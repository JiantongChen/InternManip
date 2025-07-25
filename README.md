<!-- > [!IMPORTANT]
> 🌟 Stay up to date at [openrobotlab.org.cn](https://openrobotlab.org.cn/news)! -->

![demo](assets/_static/video/internmanip_10fps.gif "demo")
<div id="top" align="left">

[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=gradio)](#)
[![doc](https://img.shields.io/badge/Document-FFA500?logo=readthedocs&logoColor=white)](https://internrobotics.github.io/user_guide/internmanip/)
[![GitHub star chart](https://img.shields.io/github/stars/InternRobotics/InternManip?style=square)](#)
[![GitHub Issues](https://img.shields.io/github/issues/InternRobotics/InternManip)](#)
<a href="https://cdn.vansin.top/taoyuan.jpg"><img src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white" height="20" style="display:inline"></a>
[![Discord](https://img.shields.io/discord/1373946774439591996?logo=discord)](https://discord.gg/5jeaQHUj4B)


</div>

# InternManip

An **All-in-one** Robot Manipulation Learning Suite for Polcy Models Training and Evaluation on Various Datasets and Benchmarks.



## 🏠 Highlights



**InternManip** provides the infrastructure for reproducing & developing the <u>state-of-the-art robot manipulation policies</u>, standardizing **🗄️dataset formats**, **⚙️model interfaces**, and **📝evaluation protocols**.


<p align="center"><b>Available Content</b></p>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
         <b>Policy Models</b>
      </td>
      <td>
         <b>Training Datasets</b>
      </td>
      <td>
         <b>Benchmarks</b>
      </td>
   </tr>
   <tr align="center" valign="top">
      <td>
         <ul>
            <li align="left"><a href="">GR00T-N1</a></li>
            <li align="left"><a href="">GR00T-N1.5</a></li>
            <li align="left"><a href="">Pi-0</a></li>
            <li align="left"><a href="">DP-CLIP</a></li>
            <li align="left"><a href="">ACT-CLIP</a></li>
            <li align="left">InternVLA-M1/A1 (coming soon...)</li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">GenManip-v1</a></li>
            <li align="left"><a href="">CALVIN</a></li>
            <li align="left"><a href="">Google-Robot</a></li>
            <li align="left"><a href="">BridgeData-v2</a></li>
            <li align="left">InternData-M1/A1 (coming soon...)</li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">GenManip-v1</a></li>
            <li align="left"><a href="">CALVIN</a></li>
            <li align="left"><a href="">Simpler-Env</a></li>
            <li align="left">InternBench-M1/A1 (coming soon...)</li>
         </ul>
      </td>
   </tbody>
</table>

### What can you do with InternManip?
- 🔄 **Reproduce state-of-the-art policy models** on popular robot manipulation datasets.
- 📊 **Train new policies** with heterogeneous policy architecture: end2end model (VLA, Action Expert) & agent framework.
- 🌍 **Flexible policy deployment** in any third-party benchmarks via a client-server setup.


### What's included?
- ✅ Unified dataset format & loaders for 4+ datasets.
- ✅ 5 pre-integrated policy models for training & evaluation.
- ✅ Standard training workflow and server-client evaluation engine.

### Why InternManip?
- 🙅🏻‍♂️ Stop re-implementing baselines. 
- 🙅🏻 Stop struggling with dataset formats. 
- 💡 Focus on *policy innovation*, not infrastructure.



## 🔥 News
- **\[2025/07\]** We are hosting 🏆IROS 2025 Grand Challenge, stay tuned at [official website](https://internrobotics.shlab.org.cn/challenge/2025/).
- **\[2025/07\]** Try the SOTA models on GenManip at [Gradio Demo](#).
- **\[2025/07\]** InternManip `v0.1.0` released, [change log](#).


## 📋 Table of Contents
- [🏠 Highlights](#-highlights)
- [🔥 News](#-news)
- [📋 Table of Contents](#-table-of-contents)
- [🚀 Getting Started](#-getting-started)
- [📚 Documentation \& Tutorial](#-documentation--tutorial-wip)
- [📦 Benchmarks \& Baselines](#-benchmarks--baselines-wip)
- [🔧 Support](#-support)
- [👥 Contribute](#-contribute)
- [🔗 Citation](#-citation)
- [📝 TODO List](#-todo-list)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)







## 🚀 Getting Started

### Prerequisites

- Ubuntu 20.04, 22.04
- CUDA 12.4 
- GPU:
The GPU requirements for model running and simulation are different, as shown in the table below:


<table align="center">
  <tbody>
    <tr align="center" valign="middle">
      <td rowspan="2">
         <b>GPU</b>
      </td>
      <td rowspan="2">
         <b>Model Training & Inference</b>
      </td>
      <td colspan="3">
         <b>Simulation</b>
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         CALVIN
      </td>
       <td>
         Simpler-Env
      </td>
       <td>
         Genmanip
      </td>
 
   </tr>
   <tr align="center" valign="middle">
      <td>
         NVIDIA RTX Series
      </td>
      <td>
         ✅
      </td>
      <td>
         ✅
      </td>
      <td>
         ✅
      </td>
      <td>
         ✅
      </td>
   </tr>
   <tr align="center" valign="middle">
      <td>
         NVIDIA V/A/H100 
      </td>
      <td>
         ✅
      </td>
      <td>
         ✅
      </td>
      <td>
         ✅
      </td>
      <td>
         ❌
      </td>
   </tr>
  </tbody>
</table>

> [!NOTE]
> We provide a flexible installation tool for users who want to use InternManip for different purposes. Users can choose to install the training and inference environment, and the individual simulation environment independently.


### Installation

We provide the installation guide [here](https://internrobotics.github.io/user_guide/internmanip/quick_start/installation). You can install locally or use docker and verify the installation easily.


## 📚 Documentation \& Tutorial (WIP)

We provide detailed [docs](https://internrobotics.github.io/user_guide/internmanip/quick_start/) for the basic usage of different modules supported in InternManip. Here are some shortcuts to common scenarios:
- [How to train and evaluate a model?](https://internrobotics.github.io/user_guide/internmanip/quick_start/train_eval)
- [How to customize your model?](https://internrobotics.github.io/user_guide/internmanip/quick_start/add_model)
- [How to import a new dataset?](https://internrobotics.github.io/user_guide/internmanip/quick_start/add_dataset)
- [How to import a new benchmark?](https://internrobotics.github.io/user_guide/internmanip/quick_start/add_benchmark)

Welcome to try and post your suggestions!


## 📦 Benchmarks & Baselines (WIP)


InternManip offers implementations of multiple manipulation policy models—**GR00T-N1**, **GR00T-N1.5**, **Pi-0**, **DP-CLIP**, and **ACT-CLIP**—as well as curated datasets including **GenManip**, **Simpler-Env**, and **CALVIN**, all organized in the standardized **LeRobot** format.

The available `${MODEL}`, `${DATASET}`, `${BENCHMARK}` and their results are summarized in the following tables:

### CALVIN (ABC-D) Benchmark
| Model  | Dataset/Benchmark | Score (Main Metric) | Model Weights |
| ------------ | ---- | ------------- | ------- | 
| `gr00t_n1` | `calvin_abcd` | | |
| `gr00t_n1_5` | `calvin_abcd` | | |
| `pi0` | `calvin_abcd` | | |
| `dp_clip` | `calvin_abcd` | | |
| `act_clip` | `calvin_abcd` | | |

### Simpler-Env Benchmark
| Model  | Dataset/Benchmark | Success Rate | Model Weights |
| ------------ | ------------- | ------------- | ------- |
| `gr00t_n1` | `google_robot` | | |
| `gr00t_n1_5` | `google_robot` | | |
| `pi0` | `google_robot` | | |
| `dp_clip` | `google_robot` | | |
| `act_clip` | `google_robot` | | |
| `gr00t_n1` | `bridgedata_v2` | | |
| `gr00t_n1_5` | `bridgedata_v2` | | |
| `pi0` | `bridgedata_v2` | | |
| `dp_clip` | `bridgedata_v2` | | |
| `act_clip` | `bridgedata_v2` | | |

### Genmanip Benchmark
| Model  | Dataset/Benchmark | Success Rate | Model Weights |
| ------------ | ------------- | ------------- | ------- |
| `gr00t_n1` | `genmanip_v1` | | |
| `gr00t_n1_5` | `genmanip_v1` | | |
| `pi0` | `genmanip_v1` | | |
| `dp_clip` | `genmanip_v1` | | |
| `act_clip` | `genmanip_v1` | | |

Please refer to the [benchmark documentation](https://internmanip.github.io/usage/get_started/run-benchmark-baseline.html) for more details on how to run the benchmarks and reproduce the results.

<!-- To fine-tune your own model or configure custom evaluations, please follow the [Getting Started]() guide. -->

## 🔧 Support

Join our [WeChat](https://cdn.vansin.top/taoyuan.jpg) support group or [Discord](https://discord.gg/5jeaQHUj4B) for any help.

## 👥 Contribute

If you would like to contribute to InternManip, please check out our [contribution guide](https://internrobotics.github.io/user_guide/internmanip/contribute).
For example, raising issues, fixing bugs in the framework, and adapting or adding new policies and data to the framework.


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@misc{internmanip2025,
    title = {InternManip: An All-in-one Robot Manipulation Learning Suite for Polcy Models Training and Evaluation on Various Datasets and Benchmarks},
    author = {InternManip Contributors},
    howpublished={\url{https://github.com/InternRobotics/InternManip}},
    year = {2025}
}
@inproceedings{gao2025genmanip,
    title={GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation},
    author={Gao, Ning and Chen, Yilun and Yang, Shuai and Chen, Xinyi and Tian, Yang and Li, Hao and Huang, Haifeng and Wang, Hanqing and Wang, Tai and Pang, Jiangmiao},
    booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
    pages={12187--12198},
    year={2025}
}
@inproceedings{grutopia,
    title={GRUtopia: Dream General Robots in a City at Scale},
    author={Wang, Hanqing and Chen, Jiahe and Huang, Wensi and Ben, Qingwei and Wang, Tai and Mi, Boyu and Huang, Tao and Zhao, Siheng and Chen, Yilun and Yang, Sizhe and Cao, Peizhou and Yu, Wenye and Ye, Zichao and Li, Jialun and Long, Junfeng and Wang, ZiRui and Wang, Huiling and Zhao, Ying and Tu, Zhongying and Qiao, Yu and Lin, Dahua and Pang Jiangmiao},
    year={2024},
    booktitle={arXiv},
}
```

</details>

## 📝 TODO List
- \[x\] Release the baseline methods, checkpoints and benchmark data.
- \[x\] Release the guidance and tutorials.
- \[ \] Polish APIs and related codes.
- \[ \] Support closed-loop evaluation.
- \[ \] Release the technical report.
- \[ \] Support online interactive training.


## 📄 License

InternManip's assets and codes are [MIT licensed](LICENSE). 
The open-sourced data are under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
Other datasets (including Calvin and Simpler) inherit their own distribution licenses.

## 👏 Acknowledgements
- [CALVIN](https://github.com/mees/calvin): A synthetic benchmark for training and evaluating robotic manipulation policies.
- [Simpler-Env](https://github.com/simpler-env/SimplerEnv): A real-sim consistent manipulation benchmark for evaluating robotic manipulation policies.
- [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T): This codebase is developed on top of the Isaac GR00T framework, with substantial restructuring and customization to better suit our experimental needs.
- [LeRobot](https://github.com/huggingface/lerobot): The data format used in this project largely follows the conventions of LeRobot.
- [InternUtopia](https://github.com/OpenRobotLab/GRUtopia) (Previously `GRUtopia`): The evaluation on GenManip relies on the InternUtopia platform.
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): We use some utilities from Orbit (Isaac Lab) for driving articulated joints in Isaac Sim.
