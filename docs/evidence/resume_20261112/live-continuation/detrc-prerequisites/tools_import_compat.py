from pathlib import Path
p=Path('/root/autodl-tmp/TempoRAC/external-baselines-20261112/detrc-prerequisites/candidate/tools/__init__.py')
p.write_text('# Copyright (c) OpenMMLab. All rights reserved.\n# Released tree has no analysis/deployment packages. Import concrete tools explicitly.\n')
