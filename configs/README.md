# Experiment configurations

JSON is used deliberately so every run has a small, inspectable configuration
without introducing a configuration framework. Configs use schema version 1 and
must include dataset, model, training, evaluation, transition, and controller
sections. Fields that do not apply to a baseline are explicit rather than hidden.

Rotation split manifests live in `data/splits/`. They contain the exact sample ID,
base scene or shape ID, render seed, angle, mirror flag, and label for each split.
The first prefix of the training split defines the shared nested data-efficiency
subsets. Regenerate the committed manifests with:

```bash
.venv/bin/python scripts/generate_rotation_manifests.py
```

Every training run writes a resolved config, run metadata, per-epoch metrics,
predictions keyed by sample ID, a best validation-selected checkpoint, and a final
summary. Test splits must never be used for checkpoint or threshold selection.
