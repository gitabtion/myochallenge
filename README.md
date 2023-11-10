# MyoChallenge 2023

This is the code repository for winning the Boading ball task of the [MyoChallenge 2023](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2023). Our team was named [CarbonSiliconAI](https://carbonsilicon.ai/)

Our team comprised:
- Heng Cai, Zhuo Yang and Wei Dang, CarbonSiliconAI

Here we have documented a summary of our approach including our key insights and intuitions along with all the training steps and hyperparameters [here](docs/summary.md).

## Software requirements

Listed in `requirements.txt`. Note that there is a version error with some packages, e.g. `stable_baselines3`, requiring later versions of `gym` which `myosuite` is incompatible with. If your package manager automatically updates gym, do a `pip install gym==0.13.0` (or equivanlent with your package manager) at the end and this should work fine. If you experience any issues, feel free to open an issue on this repository or contact us via email.

## Usage

Run `python src/main_relocate.py` to start a training. Note that this starts training from one of the pre-trained models in our curriculum. You can find all the trained models along with the scripts used to train them and the environment configurations [here](trained_models). The full information about the training process can be found in the [summary](docs/summary.md).

To evaluate the best single policy network (see the [summary](docs/summary.md)), run `python src/eval_sb3.py`.

## Reference
1. Caggiano, V., Durandau, G., Wang, H., Chiappa, A., Mathis, A., Tano, P., Patel, N., Pouget, A., Schumacher, P., Martius, G., Haeufle, D., Geng, Y., An, B., Zhong, Y., Ji, J., Chen, Y., Dong, H., Yang, Y., Siripurapu, R., Ferro Diez, L.E., Kopp, M., Patil, V., Hochreiter, S., Tassa, Y., Merel, J., Schultheis, R., Song, S., Sartori, M. &amp; Kumar, V.. (2022). MyoChallenge 2022: Learning contact-rich manipulation using a musculoskeletal hand. <i>Proceedings of the NeurIPS 2022 Competitions Track</i>, in <i>Proceedings of Machine Learning Research</i> 220:233-250 Available from https://proceedings.mlr.press/v220/caggiano22a.html.
2. https://github.com/amathislab/myochallenge

