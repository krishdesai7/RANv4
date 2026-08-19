For train.py, here's what I measured with an in-project probe (control error included, so the results are real). Both checkers behave identically on all of it:

Pattern Enforced?
JaxArray parameter, passed a str
flagged by both
as*int: int = ops.sum(a)
silent — keras.ops is fully opaque
raw = jax.jit(f), then s: str = raw(...)
silent — jit erases to Any
step: Step = jax.jit(f) with a Protocol, misused
flagged by both
So the ceiling is clear. Nothing inside an ops.* expression can ever be checked — keras ships no py.typed, so ops.sum, ops.log and ops.squeeze are Unknown to pyrefly and ty alike, and a transposition like ops.sum(raw*w * (one - y)) versus ops.sum(raw_w) \* (one - y) stays invisible. But every seam around that arithmetic is enforceable today, and the seams are where the current Any leaks out into the rest of the file.

The change I'd make, all inside train.py:

normalize_weights and weighted_bce take ArrayLike (jax's, which covers both the NumPy arrays \_run_epoch passes and the tracers inside jit) and return JaxArray. Declaring the return narrows the opaque ops result at exactly one point per function, which is what stops it spreading.
The three closures in \_make_steps lose their bare -> tuple and become -> tuple[JaxArray, Variables], with Variables on the variable-list parameters.
disc_grad_fn/gen_grad_fn stop being Callable[..., tuple]. jax.value_and_grad(..., has_aux=True) reveals as (...) -> tuple[Any, Any], so a precise alias tuple[tuple[JaxArray, Variables], Variables] is a legal narrowing and makes the destructuring at the call sites typed instead of Any.
Replace JitWrapped with two protocols, TrainStep and EvalStep, referencing TrainState directly. \_make_steps returns those, and \_run_epoch/\_eval_dataset accept them. This is the load-bearing one: today disc_step(...) returns Any, so d_losses[step] = d_loss and float(eval_step(...)) are unchecked; with the protocol they're JaxArray and the argument order gets checked too.
Two adjacent cleanups I'd fold in only if you want them: dropping cast(typ=T, val=-np.inf) for a plain float accumulator in the early-stopping comparison, and converting the epoch means to float where they're appended so history can be dict[str, list[float]] instead of list[SupportsFloat] — that matches what \_save_run and the plots actually consume.

I'd verify with pyrefly and ty on the file, pytest tests/test_train.py (which holds your float64 reduction guard), and a short real run to confirm the jitted steps still trace.

The StatelessOptimizer protocol in src/ran/rantypes/types.py named Keras's gradient parameter gradients; Keras calls it grads. So gen_step's converted keyword call type-checked fine and raised TypeError at runtime. I renamed it in the protocol and fixed the call site. Worth noting the protocol is hand-written against an untyped library, so it can only be as accurate as we make it — that mismatch was invisible to both checkers by construction. disc_step still passes those three positionally; this needs to be fixed.

The user wants the full tightening plus the two cleanups, and also wants me to evaluate jaxtyping for shape/dtype checking before deciding whether to adopt it, with everything going into rantypes/types.py. I need to be careful though, since TrainStep/EvalStep reference TrainState which actually lives in train.py, so I'll need to handle that cross-reference.

To avoid a circular import, I can import TrainState under TYPE_CHECKING only, since types.py is in rantypes and train.py imports rantypes — there's already precedent for this pattern in events.py with ArrayDataset. Since TrainState stays owned by train.py per CLAUDE.md, I'll add a TYPE_CHECKING-only import in types.py, which won't create a runtime cycle since train.py's keras import only happens at runtime, never triggered from types.py.

Now I'm defining the type aliases and Protocol classes for the gradient functions, using jax's ArrayLike for tensor-like inputs and specifying the exact positional call signatures for DiscGradFn and GenGradFn...

I'm also weighing whether two separate Protocols for the grad functions are worth the verbosity compared to a simpler generic Callable alias, since the Protocols give stricter argument checking but add more boilerplate for just two use cases.

Deciding to keep the Protocols in types.py since the wrapped grad functions currently go through an unchecked Callable, so the stricter typing is worth it. Now I'm planning the train.py edits: importing the new type names, annotating normalize_weights and weighted_bce, and continuing to annotate the rest of the functions.

I'll keep \_run_epoch returning the array type since it's meaningful, converting to float only at the history append boundary, and \_eval_dataset returns a tuple of arrays too. Then I need to export the new types from rantypes/init.py, checking how re-exports work there, and finally verify everything with pyrefly, ty, ruff, complexipy, the test suite, and a short real training run.

Checking the CLI, there's no --n-epochs flag — train_command only exposes hidden_units, n_layers, patience, seed, data_seed, n_samples, batch_size, config, dataset, variable, and load_run. So a smoke test would run up to 100 epochs with patience 5, but with small n_samples and low patience it should finish quickly. I'm considering running it with reduced samples, batch size, and patience 1, though that writes artifacts into the runs/ directory as a side effect — I should think about whether to run it from a temp directory to avoid polluting the repo.

Since the cache directory is also relative to cwd, running from /tmp would isolate both the run output and any generated dataset cache, which seems like the cleanest approach. I'll plan to cd into a temp directory and invoke the CLI there with an absolute path to the config file, keeping
