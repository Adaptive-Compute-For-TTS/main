# nfe-adapter

The **NFE Adapter** box in the architecture: decides and hands over the NFE (denoising steps per
patch) that the fixed-NFE executor (`gpu-executor/`) runs.

| Mode | Where the NFE comes from |
|---|---|
| `fixed` | the request (`runs.nfe`), unchanged from today |
| `adaptive` | the shared predictor, chosen once per request before generation: [`predictor/`](predictor/) |

Per-patch budgets (the NFE changing inside one request) are a separate track: scheduler interface and
executor/cache support, Yulchan.

- `predictor/`: inference package for the shared predictor (`predict_budget`), Tony. The bundle
  contract the training side exports to is in [`predictor/README.md`](predictor/README.md).
