# RDS-Contrastive-Model

This repository presents the implementation of the **RDS-Contrastive** model, proposed to handle noisy samples during the training of machine learning models.

The method focuses on improving robustness under label noise scenarios using a contrastive learning strategy combined with RDS mechanisms.

---

## Docker Image

You can push the Docker image using:

```bash
docker push vitorbds/tf2_llp:tagname

docker pull vitorbds/tf2_llp:tagname

## Example of Running the Model
python main_coat_rds.py \
    --noise_type pairflip \
    --dataset cifar100 \
    --batch_size 128 \
    --noise_rate 0.45 \
    --results /share_alpha_2/vitor/cifar_100_rds/pairflip_45/rodada_5

| Parameter      | Description                                         |
| -------------- | --------------------------------------------------- |
| `--noise_type` | Type of label noise (e.g., `pairflip`, `symmetric`) |
| `--dataset`    | Dataset used for training (e.g., `cifar100`,'cifar10,'mnist')        |
| `--batch_size` | Training batch size                                 |
| `--noise_rate` | Noise ratio (e.g., 0.45 = 45%)                      |
| `--results`    | Directory to save experiment results                |

