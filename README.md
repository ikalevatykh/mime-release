# MImE - Manipulation Imitation Environments

This package provides robotic manipulation [gym](https://github.com/openai/gym) environments for testing Imitation and Reinforcement algorithms. It also provides a framework to create your own environments.

One of the key features of MimE is the ability to create expert scripts to solve environments. This provides a stable source of expert trajectories and repeatability of experiments for Behavioral Cloning / Imitatation Learning.

For example of usage see [Learning policies for robotic manipulation](https://github.com/ikalevatykh/rlbc) package.


## Install

```bash
git clone https://github.com/ikalevatykh/mime-release.git
cd mime-release
pip install -r requirements.txt
python setup.py develop
```

## Citation
If you find this repository helpful, please cite our work:

```
@inproceedings{learningsim2real2019,
  author    = {Alexander Pashevich and Robin Strudel and Igor Kalevatykh and Ivan Laptev and Cordelia Schmid},
  title     = {Learning to Augment Synthetic Images for Sim2Real Policy Transfer},
  booktitle = {IROS},
  year      = {2019},
}

@inproceedings{rlbc2020,
  author    = {Robin Strudel and Alexander Pashevich and Igor Kalevatykh and Ivan Laptev and Josef Sivic and Cordelia Schmid},
  title     = {Learning to combine primitive skills: A step towards versatile robotic manipulation},
  booktitle = {ICRA},
  year      = {2020},
}
```
