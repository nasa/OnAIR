# OnAIR: Applications of The NASA On-Board Artificial Intelligence Research Platform.

Presented at the AAAI Conference on Artificial Intelligence, 2025.
Published as a special topic article in AI magazine available here: https://doi.org/10.1002/aaai.70020

## Abstract

Infusing artificial intelligence algorithms into production aerospace systems can be challenging due to costs, timelines, and a risk-averse industry.
We introduce the Onboard Artificial Intelligence Research (OnAIR) platform, an open-source software pipeline and cognitive architecture tool that enables full life cycle AI research for on-board intelligent systems.
We begin with a description and user walk-through of the OnAIR tool.
Next, we describe four use cases of OnAIR for both research and deployed onboard applications, detailing their use of OnAIR and the benefits it provided to the development and function of each respective scenario.
We conclude with remarks on future work, future planned deployments, and goals for the forward progression of OnAIR as a tool to enable a larger AI and aerospace research community.

# Standards and Schematics for Intelligent Extensible Mission Architectures in Space

Presented at the 2nd Distributed Autonomy for Space Systems Workshop, 2025.
Paper available here: https://ntrs.nasa.gov/api/citations/20250005873/downloads/DASS-Camera-Ready.pdf

## Abstract

As the space sector trends toward complex mission types, the demand for multi-asset, multi-generational, and multiorganizational paradigms has grown.
It is critical that the spaceflight community builds the infrastructure needed to realize these mission architecture goals.
To this end, we present a standard and schematic for intelligent extensible mission architectures, which will allow missions of the future to be distributed, heterogeneous, incremental, and interoperable for enhanced flexibility, adaptability, and responsiveness in space.
We begin by describing a motivation for intelligent extensibility, followed by a review of related work in standards and autonomous multi-agent systems.
Next, we present the theoretical definition and schematic of our standard, followed by an illustrative example and description of experimental results.
We conclude with suggestions for adoption of our standards in future work.

# Evaluation and Integration of YOLO Models for Autonomous Crater Detection

Presented at the IEEE AeroSpace Conference, 2025.
Paper available here: https://ntrs.nasa.gov/api/citations/20240015360/downloads/William_IEEE_Aero_Paper.pdf

## Abstract

Advancements in deep learning and computer vision are enabling the development of expanded spacecraft capabilities.
One field of interest is automated crater detection which has applications in terrain relative navigation, pose estimation, and planetary science.
While continued development of learning-based crater detection algorithms (CDAs) has led to more accurate and performant models, there is currently a limited discussion on how these models might be integrated into future mission infrastructure.
Specifically, we identify the deployment of CDA models onto resource-constrained, flight-like hardware and the interaction of CDAs with existing flight software as key areas of investigation.
To this end, we first introduce a novel Lunar crater dataset based on digital elevation map (DEM) data and 1.2 million known crater positions, leveraging the Blender 3D software to render surface imagery with ground truth bounding box labels.
We evaluate the You Only Look Once (YOLO) family of models on this dataset for crater recognition performance while providing runtime and memory analysis on representative flight hardware, consisting of a Teledyne radiation-tolerant LS1046-Space CPU and a Google Coral Edge TPU accelerator.
We comment on the choice of activation function in the YOLO architecture as it relates to detection performance and inference time.
Carefully considering model operations is essential because Edge TPU compatibility is paramount for near-realtime, onboard deep learning execution.
Finally, we put forth an example implementation of YOLO CDA within the On-Board
Artificial Intelligence Research (OnAIR) platform, a cognitive architecture for autonomous applications that can interact with flight software frameworks such as NASA’s core Flight System (cFS).

# The Onboard Artificial Intelligence Research (OnAIR) Platform

Presented at AI in and for SPACE (SPAICE) 2024.
Paper available here: https://zenodo.org/records/13885517

## Abstract

In this paper, we present the NASA On-Board Artificial Intelligence Research (OnAIR) Platform, a dual-use tool for rapid prototyping autonomous capabilities for earth and space missions, serving as both a cognitive architecture and a software pipeline.
OnAIR has been used for autonomous reasoning in applications spanning various domains and implementation environments, supporting the use of raw data files, simulators, embodied agents, and recently in an onboard experimental flight payload.
We review the OnAIR architecture and recent applications of OnAIR for autonomous reasoning in various projects at NASA, concluding with a discussion on the intended use for the public domain, and directions for future work.
