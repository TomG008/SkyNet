# SkyNet
This is a repository for SkyNet, a lightweight DNN specialized in object detection. SkyNet has demonstrated on [the 56th IEEE/ACM Design Automation Conference System Design Contest (DAC-SDC)](http://www.cse.cuhk.edu.hk/~byu/2019-DAC-SDC/index.html) by helping our GPU team, iSmart3-SkyNet, and FPGA team, iSmart3, win the first place winner award for both GPU and FPGA tracks. It can deliver 0.7312 Intersection over Union (IoU), 67.33 frames per second (FPS) and 0.7160 IoU, 25.05 FPS using TX2 GPU and Ultra96 FPGA, respectively.

The team members of iSmart3-SkyNet are: Xiaofan Zhang*, Haoming Lu*, Jiachen Li, Cong Hao, Yuchen Fan, Yuhong
Li, Sitao Huang, Bowen Cheng, Yunchao Wei, Thomas Huang, Jinjun Xiong, Honghui Shi, Wen-mei Hwu, Deming Chen. 
(*equal contributors)


## Platform
Jetson Tx2, Jetpack 4.2

## Install
```
$ sudo bash install.sh
```
## Test on given dataset
```
$ python3 run.py
```
The dataset is supposed to be organized as [required](https://d1b10bmlvqabco.cloudfront.net/attach/jrckw1628ejd9/jux80pibriz3qy/jvlmoykue8qf/Submission_requirement.txt).
## Run the demo (webcam)
```
$ python3 demo.py
```
## References
If you find SkyNet useful, please cite the [SkyNet paper](https://arxiv.org/abs/1906.10327):
```
@article{zhang2019skynet,
  title={SkyNet: A Champion Model for {DAC-SDC} on Low Power Object Detection},
  author={Zhang, Xiaofan and Li, Yuhong and Hao, Cong and Rupnow, Kyle and Xiong, Jinjun and Hwu, Wen-mei and Chen, Deming},
  journal={arXiv preprint arXiv:1906.10327},
  year={2019}
}
```
More details regarding the SkyNet design motivations and SkyNet FPGA accelerator design can be found in our [ICML'19 workshop paper](https://arxiv.org/abs/1905.08369) and the [DAC'19 paper](https://arxiv.org/abs/1904.04421), respectively.
```
@article{zhang2019bi,
  title={A Bi-Directional Co-Design Approach to Enable Deep Learning on {IoT} Devices},
  author={Zhang, Xiaofan and Hao, Cong and Li, Yuhong and Chen, Yao and Xiong, Jinjun and Hwu, Wen-mei and Chen, Deming},
  journal={arXiv preprint arXiv:1905.08369},
  year={2019}
}
```
```
@inproceedings{hao2019fpga,
  title={{FPGA/DNN} Co-Design: An Efficient Design Methodology for {IoT} Intelligence on the Edge},
  author={Hao, Cong and Zhang, Xiaofan and Li, Yuhong and Huang, Sitao and Xiong, Jinjun and Rupnow, Kyle and Hwu, Wen-mei and Chen, Deming},
  booktitle={Proceedings of the 56th Annual Design Automation Conference},
  pages={206},
  year={2019},
  organization={ACM}
}
```
