# MotionBERT generalization result binding audit — 2026-08-18

## Outcome

The historical `seed20260725/unseen_person_count/prototype` result was
independently replayed from the retained checkpoint, frozen Git source, and
three protocol datasets. The replay artifacts are byte-identical to the
historical artifacts. The reported test metric is therefore reproducibly bound
to the inputs below.

The result is:

- video-level relative MAE (`mae`): `0.08750410332352425`;
- video-level OBO: `0.175`;
- per-video mean person-relative MAE (`avg_mae`): `0.16792673992673993`;
- per-video mean person OBO (`avg_obo`): `0.725`;
- per-person raw count MAE (`legacy_person_mae`): `2.0125`.

The implementation calls the first quantity `mae`, but computes it as the
mean of `abs(predicted_total - ground_truth_total) / ground_truth_total` over
videos. It is a relative MAE/NMAE quantity, not raw count MAE.

The MPCounter paper-facing OBO column is `avg_obo`: for each person it tests
whether the absolute count error is at most one, averages those indicators
within each video, and then averages over videos. Under that definition this
replayed result is `0.725`, above the requested `0.6`. The stricter `obo=0.175`
instead tests whether the whole-video total-count error is at most one.

## Frozen source and checkpoint

- MotionBERT Git commit:
  `984e3ac491fb873b92818a0df81304ac9d90a45e`
- checkpoint path:
  `/media/lenovo/data2/shenjihe/mjx/MotionBERT-main/MotionBERT-main/checkpoint/latest_epoch.bin`
- checkpoint SHA-256:
  `737fbaa9c9ad524eee4e3c3bba8a0d86e2bc978b494bea78561d7c125cfc97d9`
- inference entrypoint SHA-256:
  `21097ccdc654685f01e22955f54c3e0ccf2050ae312d637cbb3247802207b7fb`
- suite entrypoint SHA-256:
  `d90c1c3c909e336e70d5ec1f31cd38982d4f06d30e915979e8e8ed94e40ac405`

The replay used one visible CUDA device and the suite's original inference
arguments: `tiny_period_boundary`, temporal width 96, kinematic features,
multi-scale TCN, FFT count prior, maximum length 320, batch size 4, and zero
data-loader workers.

## Dataset bindings

- train pickle SHA-256:
  `fa40bbe6d5fe3ea991431fcd4d21c7e2a00ed98b91ea4e028be44f938f74a2d6`
- validation pickle SHA-256:
  `c6841b46b712a00b7097393326a17eebcaeb83d50dc1cb2359a36082f206df95`
- test pickle SHA-256:
  `198ad0ed5a065ebbe8f327cb53a328d5b416c3e1b2ab18db9749c82ed8764970`

The protocol reports empty train/validation/test source-ID intersections. The
prototype calibrator uses source train labels for selection and source
train+validation labels for its final nearest-neighbour reference bank. Target
test labels are used only for the final evaluation.

## Byte-identical replay evidence

- train base predictions SHA-256:
  `81672daef5018ba2ee21d187af1ca62c398129c55dbd969c85c243337b394b53`
- validation base predictions SHA-256:
  `374a6752035a7de6c7580e7aee5facbcd04226b5a7b972d592a86118002389ef`
- test base predictions SHA-256:
  `b9a01cdba59d9b12797f1cbbd6ff3cabbb8cdd62db2b29def3291913459e6b05`
- prototype metrics SHA-256:
  `3c344ff8c3c731e2d243c9d74293b17ddd77b5a7b74ee040a3f09ccac3f8f8f4`
- prototype calibrator SHA-256:
  `b86f123dd3dbf7a7585feb78b69d75493e20bf432c6cd0e26ab2ffc44c101a61`
- test calibrated predictions SHA-256:
  `44fb21eb51774d07639a6e26e71f1cb94d317c28245bddd58a0a8f02e9f9ab49`
- validation calibrated predictions SHA-256:
  `f2adda97ff82f1aa538de4ff29aa82e484496b336c06c2b564292d5dce7120bb`
- test nearest-neighbour evidence SHA-256:
  `6930e3d30103ff11c90e4ddfb457e8d6e83d010feca2f80299cfc6741e35981b`

Historical and replayed copies have the same SHA-256 for every artifact listed
above.

## Replay location

The detached source checkout is:

`/media/lenovo/data2/pams-rac/audits/motionbert-replay-984e3ac`

The independent replay outputs are:

`/media/lenovo/data2/pams-rac/audits/motionbert-binding-984e3ac-seed20260725-unseen-person-count`

## Claim boundary

This is a reproducible, supervised prototype-calibration result for the
MPCounter unseen-person-count generalization protocol. It may be reported under
that protocol with the relative-MAE definition above. It is not a raw MAE of
`0.0875`, is not a UCFRep dev84 result, and is not a self-supervised TempoRAC
checkpoint result. TempoRAC comparisons must either reproduce this exact
protocol and metric or label it as an external, non-comparable baseline.

The `0.725` AvgOBO is the result for protocol seed `20260725`. The corresponding
prototype AvgOBO values for seeds `20260723`, `20260724`, and `20260725` are
`0.06875`, `0.30625`, and `0.725`; their unweighted mean is approximately
`0.36667`. Consequently, `0.725` is a byte-reproducible single-protocol-split
result, not evidence that a three-seed mean exceeds `0.6`.
