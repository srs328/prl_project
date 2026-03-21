---
url: https://hpc.umassmed.edu/doc/index.php?title=Running_Computations
title: "Running Computations - UMass Chan SCI Cluster"
date: 2026-03-20T23:31:00.922Z
lang: en-US
---

[UMass Chan SCI Cluster UMass Chan SCI Cluster](/doc/index.php?title=Main_Page "Visit the main page")

[Log in](/doc/index.php?title=Special:UserLogin&returnto=Running+Computations)↓

[Personal tools](#)

*   [Log in](/doc/index.php?title=Special:UserLogin&returnto=Running+Computations "You are encouraged to log in; however, it is not mandatory [alt-shift-o]")

# Running Computations

From UMass Chan SCI Cluster

## Contents

*   [1 Getting Started](#Getting_Started)
    *   [1.1 Submitting your first job](#Submitting_your_first_job)
*   [2 Please Note](#Please_Note)
*   [3 Basics](#Basics)
*   [4 Interactive vs Batch](#Interactive_vs_Batch)
*   [5 Command line vs Script](#Command_line_vs_Script)
*   [6 Queues](#Queues)
*   [7 Requesting cores, nodes, memory](#Requesting_cores,_nodes,_memory)
*   [8 CPU core allocation and LSF span settings](#CPU_core_allocation_and_LSF_span_settings)
    *   [8.1 Common Span Options](#Common_Span_Options)
        *   [8.1.1 span\[hosts=1\]](#span[hosts=1])
        *   [8.1.2 span\[hosts=-1\]](#span[hosts=-1])
        *   [8.1.3 span\[ptile=X\]](#span[ptile=X])
        *   [8.1.4 span\[block=X\]](#span[block=X])
    *   [8.2 Which Should You Use?](#Which_Should_You_Use?)
    *   [8.3 Key Takeaway](#Key_Takeaway)
*   [9 Requesting Job Run Times](#Requesting_Job_Run_Times)
*   [10 Intel vs AMD](#Intel_vs_AMD)
*   [11 Using GPUs](#Using_GPUs)
*   [12 Output to Files](#Output_to_Files)
*   [13 Send Output via Email](#Send_Output_via_Email)
*   [14 Additional job submission options](#Additional_job_submission_options)
*   [15 Showing and Terminating Jobs](#Showing_and_Terminating_Jobs)
*   [16 Currently available Open OnDemand Tools](#Currently_available_Open_OnDemand_Tools)

# Getting Started

You've activated an account, registered your ssh key, and can now ssh into the cluster head nodes. What next?

The first important thing to note is that logging into the head nodes (hpcc03 and hpcc04) and having a shell prompt does not mean when you run commands they are running on the cluster. The cluster is made up of a number of different machines with various resources, and in order to run your computations, you will need to first request resources, and to do this you will submit a job.

## Submitting your first job

For your first job, let's submit an interactive job. For this example we'll assume your login name is jane.doe-umw and you've logged into the head node hpcc04. Try running the command in bold below, you should see similar output when you run it.

\[jane.doe-umw@hpcc04 ~\]$ **bsub -Is -q interactive -R "rusage\[mem=4G\]" /bin/bash**
WARN: Job does not specify number of cores.  Setting 'bsub -n 1' (single core).
Job <685957> is submitted to queue <interactive>.
<<Waiting for dispatch ...>>
<<Starting on c6525c10>>
\[jane.doe-umw@c6525c10 ~\]$

The default command prompt will change from **\[_username_@hpcc03 ~\]$** or **\[_username_@hpcc04 ~\]$** to the name of a cluster processing node your job has been dispatched to. Now your session is using resources you requested from the scheduler, in this case you have a single cpu core and 4gb of memory available for use.

Let's say you have a file and you need to run some processing commands on it, now that you're in an interactive shell this is the place to run them. For example:

\[jane.doe-umw@c6525c10 ~\]$ strings randomdata | wc -l
6637425
\[jane.doe-umw@c6525c10 ~\]$ wc -l randomdata
2098997 randomdata
\[jane.doe-umw@c6525c10 ~\]$ md5sum randomdata
77e0e60b2f20088bd8bebb796bb3607c  randomdata

# Please Note

**While the SCI cluster head nodes hpcc03 and hpcc04 are where your initial shell session will run on the cluster, they are intended for users to submit jobs, transfer files, and perform administrative tasks, and should not be used for running scientific applications or calculations.**

_Limitations set on the head nodes will prevent some applications from working correctly_; when you wish to run a calculation or test software, please be sure to do so from the [Open OnDemand web portal](https://ood.umassmed.edu) or by submitting an interactive bsub job, e.g.:

  

# Basics

The SCI Cluster makes use of IBM's [LSF Spectrum Job Scheduler](/doc/index.php?title=LSF_Spectrum_Job_Scheduler "LSF Spectrum Job Scheduler"). Jobs [LSF Queues](/doc/index.php?title=LSF_Queues "LSF Queues")

# Interactive vs Batch

In LSF every job will fall into one of two categories:

*   **Interactive**
    *   Interactive jobs are jobs that maintain a connection to your shell once you've run the bsub command, allowing you to provide input and see output from the shell you submitted the job from. Any job that is not, or can not be, fully scripted out is probably a job you want to run interactively. Typically interactive jobs are used for testing things, compiling software, and running bulk processing.
    *   In order to submit an interactive job you must include the bsub option **`-Is`**.
    *   The **interactive** queue is intended for use by interactive jobs, but users may run interactive jobs in other queues as well - however, keep in mind that the **interactive** queue has the highest priority to minimize the wait time, other queues are likely to take significantly longer to start jobs whether they are interactive or batch.
    *   Here is an example interactive job a user could submit requesting 3 cpu cores, and 6GB of memory for 8 hours in the **interactive** queue:

$ bsub -Is -q interactive -W 8:00 -R "rusage\[mem=2G\] span\[hosts=1\]" /bin/bash

*   **Batch**
    *   Batch jobs (aka "non-interactive") are jobs that will run with no human intervention. The majority of jobs run on the SCI cluster are expected to be non-interactive. Once a batch job has been dispatched and starts running the only interaction a user can have with it is to pause or terminate the job.
    *   All jobs, by default, are batch jobs, unless the user uses the **`-Is`** option with the bsub command.
    *   Unless you specify otherwise, when you submit a batch job, the output from the job will be emailed to you once the job has ended.
    *   You can use the bsub "-o" and "-e" options to specify output files for job output and errors to be redirected to instead of being emailed to you. e.g.:

bsub -q long -o my\_out.%J -e my\_err.%J ./script

This will write the output to two different files, where the %J in each file name is replaced with the jobid number that the job was assigned. You can check the contents of these files as the job runs, or after it has finished, to see the output.

# Command line vs Script

In addition to submitting a job as a single line from the shell prompt, you can also create a script to submit to LSF. LSF will accept normal scripts with bsub, but it also has the option to specify job parameters in the script. For example:

**myscript.sh**

#!/bin/sh
#BSUB -n 1
#BSUB -R "rusage\[mem=2G\]"
#BSUB -q long
#BSUB -W 720:00
#BSUB -o "$HOME/%J.out"
#BSUB -e "$HOME/%J.err"

_Commands to run are placed here_

To submit this script, named _myscript.sh_ to the scheduler, you would just run:

**bsub < myscript.sh**

Each line in the script starting with #BSUB tells LSF that what follows is a paramter that it should use for bsub, just as if it had been included on the bsub command line. In the case of the example above:

*   **\-n 1** tells the scheduler this job only needs one cpu core
*   **\-R "rusage\[mem=2G\]"** tells the scheduler to allocate 2G of memory for each cpu core allocated
*   **\-q long** tells the scheduler the job should be submitted to the _long_ queue (see [here](#Queues) for a list of queues)
*   **\-W 720:00** tells the scheduler the job may run for up to 720 hours/30 days, which is the maximum job length currently permitted in the long queue.
*   **\-o "$HOME/%J.out"** tells LSF the job's standard output should be written to a file in the user's home directory which is named with the jobid (which is assigned when the job is accepted by the scheduler) followed by the extension '.out'.
*   **\-e "$HOME/%J.out"** tells LSF the job's standard output should be written to a file in the user's home directory which is named with the jobid followed by the extension '.err'.
*   _**Commands to run are placed here**_ would be replaced by the body of your normal shell script, running whatever commands you want the job to execute.

# Queues

As of this writing the available queues on the SCI Cluster are:

*   **interactive**
    *   Jobs in the interactive queue have a maximum runtime of 8 hours.
    *   Users are limited to 16 cores running in the interactive queue at a time.
    *   Interactive queue jobs have the highest priority.
*   **large**
    *   Jobs in the large queue have a maximum runtime of 96 hours.
    *   Users are limited to 800 cores running in the large queue at a time.
    *   Large queue jobs have high priority.
    *   Large queue jobs only run exclusivley on Intel nodes which have 40 cores and ~370GB of memory.
    *   Please do not specify a memory request value when submitting jobs to the large queue, they are automatically set to request the maximum memory.
*   **short**
    *   Jobs in the short queue have a maximum runtime of 8 hours.
    *   Users are limited to 512 cores running in the short queue at a time.
    *   Short queue jobs have medium priority.
*   **long**
    *   Jobs in the long queue have a maximum runtime of 720 hours (30 days).
    *   Users are limited to 1500 cores running in the long queue at a time.
    *   Long queue jobs have low priority.
*   **gpu**
    *   Please note that the gpu queue does allow interactive jobs; if you want an interactive job with one or more gpu devices, submit using 'bsub -Is -q gpu', do not use the 'interactive' queue.
    *   Jobs in the gpu queue have a maximum runtime of 720 hours (30 days).
    *   Currently users are limited to a maximum of 12 gpu devices in use at once on the SCI Cluster.
    *   If you do not specify otherwise, the default GPU settings for jobs in the 'gpu' queue are: **"num=1:mode=exclusive\_process:mps=yes:j\_exclusive=yes:gvendor=nvidia"**
    *   The GPU queue does not share nodes with other queues.

# Requesting cores, nodes, memory

*   To specify the number of cores your job will need, use the bsub **\-n** option, e.g. "bsub -n1" will request one core.
    *   If you are specifying more than one core, you will most likely want to specify that all cores need to be on the same node. You can do this with the bsub **\-R "span\[hosts=1\]"** option. e.g.: **bsub -n8 -R "span\[hosts=1\]"** will request 8 cpu cores on a single node.

*   To specify memory, use the bsub **\-R rusage\[mem=X\]** option. This will request X memory **per cpu core**. As an example, if you ran: **bsub -n8 -R "span\[hosts=1\] rusage\[mem=8G\]"** then the scheduler with start a job on a node with 8 cpu cores and 64G (8 cores x 8G per core) of memory.

If you do not specify otherwise when using bsub, the LSF scheduler will assign your job 1 core, and 1G of memory per core.

# CPU core allocation and LSF span settings

When you submit a job that uses more than one core, LSF needs to decide **where those cores come from**. By default, if you don’t tell it otherwise, LSF will place cores _anywhere it finds available slots_ across the cluster.

On our system, we’ve now added a rule:

*   If you request **under 128 cores** and don’t specify a span parameter, your job will be forced to run on a **single host** (_span\[hosts=1\]_).
*   This is because most jobs (except MPI or other distributed workloads) don’t benefit from spreading across multiple nodes; this avoids confusing behavior, and will help ensure users get the resources they expect.

If you’re running software that **should** use multiple nodes, you need to tell LSF explicitly how to place your cores using a **span string**. Typically spanning multiple nodes is **only** used when running MPI applications, which are specifically designed to be able to use more than one computer at a time.

## Common Span Options

### span\[hosts=1\]

*   All cores for your job are placed on **one host**.
*   Best for threaded applications or jobs that need shared memory.
*   Example:

 `bsub -n 32 -R "span[hosts=1]" myjob`

### span\[hosts=-1\]

*   Disables the single‑host restriction.
*   LSF can spread your cores across **any available hosts**.
*   Use this if you want a large job to run across multiple nodes.
*   Example:

 `bsub -n 200 -R "span[hosts=-1]" myjob`

### span\[ptile=X\]

*   Ensures exactly **X cores per host** are allocated.
*   Useful for MPI jobs where you want a fixed number of ranks per node.
*   Example:

 `bsub -n 64 -R "span[ptile=8]" mpirun myjob`
 → 8 cores per host, spread across 8 hosts.

### span\[block=X\]

*   Allocates cores in **blocks of size X**, packing as many blocks as possible on each host before moving to the next.
*   Good for applications that work in groups of threads or ranks.
*   Example:

 `bsub -n 32 -R "span[block=8]" myjob`
 → Allocates cores in groups of 8.

## Which Should You Use?

*   **Single‑node threaded jobs** (MATLAB, R, Python, etc.):

 → No need to set span, defaults to _hosts=1_.

*   **MPI jobs across multiple nodes:**

 → Use _span\[ptile=X\]_ to control ranks per node, or _hosts=-1_ if you just want LSF to spread them.

*   **Jobs with natural block sizes:**

 → Use _span\[block=X\]_ to pack cores in groups.

## Key Takeaway

If you don’t specify span, we’ll assume you want **all cores on one host** (up to 128 cores). If your software is designed to run across multiple nodes, **always add a span string** so LSF places your cores the way you expect.

# Requesting Job Run Times

*   To specify job run time, use the bsub **\-W** option, e.g.: **bsub -W 5:00** requests a five hour runtime, **bsub -W 720:00** requests a 720 hour (30 day) runtime, which is the longest runtime available. Please see \[Running\_Computations#Queues\] for information about individual queues and how long their runtimes can be.

# Intel vs AMD

For the SCI cluster we have configured host groups to distinguish between Intel and AMD nodes, they are **INTEL** and **AMD** respectively. If you wish to submit a job that will use only Intel, or only AMD nodes, you can specify the group with the **bsub -m** option:

bsub -m INTEL

Will only run on Intel nodes.

bsub -m AMD

Will only run on AMD nodes.

Please note: at the time of this writing the V100 GPU nodes are all Intel, and the A100 GPU node is AMD, so using the group specification incorrectly with gpu jobs may prevent a job from being able to be dispatched.

# Using GPUs

At the time of this writing we have 10 gpu nodes with 4 V100 gpu devices each, and one gpu node with 4 A100 devices. When submitting a gpu job if you want to only use specific devices the easiest way is to use the bsub "-m" option, either with **"bsub -m V100"** or **"bsub -m A100"**.

Other gpu-specific options can be set using the [\-gpu flag and options](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=o-gpu) with bsub. The most commonly used options would be:

bsub -q gpu -gpu "num=**X**"

This specifies the number of gpu devices desired per node the job uses, where X is a number that can range from 1 to 4.

# Output to Files

When submitting a job with bsub you can specify that output from the job should be written to a file:

**bsub -o _output\_file_**

The -o bsub option will write the job's standard output to the file you specify.

**bsub -e _error\_file_**

The -e bsub option will write the job's standard error to the file you specify.

If you use the special character %J in the name of either file, then %J is replaced by the job ID of the job. If you use the special character %I in the name of the error file, then %I is replaced by the index of the job in the array if the job is a member of an array. Otherwise, %I is replaced by 0 (zero).

The HPC team recommends that you specify at least an output file name with '%J' when submitting jobs so the output will be recorded; using %J (and %I when running array jobs) will prevent subsequent jobs from overwriting the output from earlier jobs.

If the specified _output\_file_ or _error\_file_ path is not accessible, the output will not be stored.

# Send Output via Email

If you do not specify output files when you submit a job, the scheduler will attempt to send the output to your UMass Chan email account.

You can choose to send output via email to a different address with the **bsub -u _email\_address_** option.

# Additional job submission options

The bsub command has many [additional options](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=bsub-options), please email us at hpc@umassmed.edu for assistance with other specifics of job submissions.

# Showing and Terminating Jobs

The **bjobs** command will show you your currently pending and running jobs.

*   Adding the **\-a** flag will also show recently ended jobs.
*   Adding the **\-p** flag will show pending jobs and what they are waiting for to be dispatched.

The **bkill** command allows you to terminate jobs.

*   **bkill _jobid_** will kill that specific jobid number.

# Currently available Open OnDemand Tools

*   An [xfce4](https://en.wikipedia.org/wiki/Xfce) graphical desktop, suitable for running applications and tools that display graphics via [X11](https://en.wikipedia.org/wiki/X_Window_System), as well as running command-line terminal windows.
*   **[MATLAB](https://www.mathworks.com/)**
*   **[Jupyter Notebook](https://jupyter.org/)**
*   **[Rstudio](https://en.wikipedia.org/wiki/RStudio)**
*   **[VSCode IDE/Editor](https://coder.com/)** which uses open-source codeserver to run VSCode-like operations on cluster nodes.
*   **[Tensorboard](https://www.tensorflow.org/tensorboard)** provides the visualization and tooling needed for machine learning experimentation.
*   **[AlphaFold 2 and 3](https://deepmind.google/science/alphafold/)** forms to submit non-interactive AlphaFold jobs.
*   **[VSCode Tunnels](https://code.visualstudio.com/docs/remote/tunnels)** to run VSCode using cluster resources. (Please do not just connect to head nodes with VS Code Remote SSH)
*   A web-based [file manager](https://en.wikipedia.org/wiki/File_manager).

Retrieved from "[https://hpc.umassmed.edu/doc/index.php?title=Running\_Computations&oldid=859](https://hpc.umassmed.edu/doc/index.php?title=Running_Computations&oldid=859)"

*   [![Powered by MediaWiki](/doc/resources/assets/poweredby_mediawiki_88x31.png)](https://www.mediawiki.org/)

*   This page was last edited on 3 December 2025, at 18:39.

*   [Privacy policy](/doc/index.php?title=SCI_Cluster:Privacy_policy)
*   [About UMass Chan SCI Cluster](/doc/index.php?title=SCI_Cluster:About)
*   [Disclaimers](/doc/index.php?title=SCI_Cluster:General_disclaimer)