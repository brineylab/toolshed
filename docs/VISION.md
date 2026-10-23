# The toolshed

Status: approved 2026-10-23, from the flyloft toolshed brainstorm. Change only with the lab.

## What it is

Most of the effort in testing an idea is not the idea. It is getting a published model to run on the lab's hardware, writing the pipeline around it, converting one tool's output into another's input, and making sure the thing that ran is the thing the paper described. The toolshed is where that effort is kept so it is paid once.

A **tool** is a capability anyone in the lab, or any agent working for them, can call: a protein language model brought up with its weights, a structure predictor, a molecular dynamics pipeline, an evaluation harness. Each tool is one container image that holds its environment, its dependencies, and its weights, and one prose skill that says what it is for, how to call it, what it needs, and what to watch out for. The image runs the same way on a laptop under podman, on a Slurm cluster under pyxis, and inside a flyloft campaign. The caller supplies the command.

The toolshed belongs to the lab, not to a project and not to flyloft. flyloft is one user of it and one contributor to it, among others. A researcher's own coding agent reads the same skills and runs the same images. Another lab forks the repository and has its own shed and its own registry with no edits.

## Why

- **The fifth idea a lab tests should take less of anyone's effort than the first.** Bringing up a model is done once, by whoever needs it first, and is there for every later question.
- **A tool version is one immutable thing.** Weights are in the image, fetched at build time and never at run time, so a result can be reproduced whatever the model's publisher does later, and caching the image caches the weights.
- **Nothing in a tool is specific to any one caller.** No flyloft-only interface, no Python-only interface: an image, a program on its PATH, files in and files out.
- **Prose over schemas, until prose proves insufficient.** A skill is what a coding agent reads. Declared parameter lists come when agents are seen misusing commands from prose, not before.

## How it grows

A tool is built because a question required it, not because it might be useful, and it is kept because it is used. Every tool is tested where it will be used, on a tiny input with a known answer, and reviewed by someone other than its builder before it is published. A tool nobody calls, or whose test no longer passes, is retired rather than left to mislead.

Tools arrive two ways. A person builds one with their own agent, following the guidance in this repository, and opens a pull request. A flyloft campaign that needs a tool the shed lacks builds it the same way: its implementer writes the tool, flyloft opens the pull request and runs the test as one of the campaign's runs, a critic reviews it, and flyloft merges. Either way the shed's CI builds the image and publishes the version.

## What it is not

- **Datasets.** Where the lab's data lives, how it is versioned, and how it is fetched is a companion to the shed, not part of it.
- **Analysis helpers.** A plotting routine or a data loader is a project's code, not a tool.
- **A workflow engine.** The shed holds capabilities. Sequencing them is the caller's job: a researcher's agent, a Slurm script, or a flyloft campaign.
