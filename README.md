# YOLO-LGA
We propose a lightweight CNN and transformer union detection model based on YOLOv5, called **YOLO-LGA**, to explore short-term dependencies and long-term dependencies in the image with CNN and transformer respectively for multi-class insect pest detection.
In order to thoroughly couple convolutional local features and transformer-based global features, this study first designs a novel local-to-global attention block, dubbed LGA which contains a local feature extractor with C3 block, a global feature extractor with Swin Transformer Block, a local-to-global cross attention module with Cross Window Transformer Block and an aggregation module.  
![LGA Block](https://github.com/zjfgh2015/DelphiAgent/blob/main/DelphiAgent%20Framework.jpg)
Then, we propose a lightweight CNN and transformer union detection model based on YOLOv5-S, called YOLO-LGA, core of which is that almost all the C3 (Cross Stage Partial Bottleneck with 3 convolutions) blocks (except the first two C3 blocks) in original YOLOv5-S are replaced by our designed LGA blocks.
![YOLO-LGA Framework](https://github.com/zjfgh2015/DelphiAgent/blob/main/DelphiAgent%20Framework.jpg)

## Installation

Install the required package using pip:

```bash
pip install requirements.txt
```
## Datasets

- For the IP102 datasets, please refer to the GitHub repository: [GitHub Repository]([https://github.com/xpwu95/IP102])

## Pretrained Checkpoints

-The final model weights file path is:
```bash
runs/train/exp/weights/best.pt
```
## Citation
If you use our framework in your research, please cite our work. 
