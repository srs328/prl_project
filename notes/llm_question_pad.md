# LLM Quetion Pad

Scratch document for writing up questions and tasks for llm agents

---

Changing the DiceCE loss weights was huge. Changing whether to include background or not doesnt seem to do much. I screwed up the num_crops_per_image; it is actually 2 in every single run. This had to do with a default setting in the segresnet templates that override certain settings. So effectively, every run had: num_crops_per_image: 2 and batch_size: 1. Just so you have context: segresnet does the following to set the hyperparameters:

```python
roi_size, levels, init_filters, batch_size = auto_adjust_network_settings(...)
```

This overrides whatever setting the user passes unless the user also sets `auto_scale_batch: false`. Then, it swaps whatever the user set for num_crops_per_image (unless the user sets auto_scale_allowed: false).

```python
if config["crop_mode"] == "ratio":
    config["num_crops_per_image"] = config["batch_size"]
    config["batch_size"] = 1
```

I'm wondering, should I bother redoing every run from run5 - run 12 with the proper settings? Or just do it for a subset of those: like one option would be to skip runs with no background. The reason is the HPC limits the number of concurrent GPU tasks, so I wonder if there are other things I could be trying instead of making sure all 8 of these runs are corrected. What do you think?

Here is the key for the loss column: bg: include_background=true; nowt means DiceCELoss(weight=None), and wt115 means DiceCELoss(weight=[1,1,5])



ok can you just do a double check that my stage3 parameter sweep (/media/smbshare/srs-9/prl_project/training/roi_train2/stage3_numcrops_bkd_constwt115) will work now? stage 2 should have had 2 crops per batch in every run. so for this one i want to use 1 and 4. I understand that first, batch_size is set from num_images_per_batch, and as long as auto_scale_batch is false (which I set in monai_config.jsonc), it will stay that way. Then num_crops_per_image will become batch_size and batch_size will become 1. is that what will happen?



### Run failed

<frozen importlib._bootstrap_external>:1324: FutureWarning: The cuda.cudart module is deprecated and will be removed in a future release, please switch to use the cuda.bindings.runtime module instead.
Traceback (most recent call last):
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/bundle/reference_resolver.py", line 155, in _resolve_one_item
    look_up_option(d, self.items, print_all_options=False)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/utils/module.py", line 136, in look_up_option
    raise ValueError(
    ...<3 lines>...
    )
ValueError: By 'sigmoid', did you mean '0::sigmoid'?
'sigmoid' is not a valid value.


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/media/smbshare/srs-9/prl_project/training/roi_train2_t1/run_losswt115_numcrop2/segresnet_0/scripts/train.py", line 27, in <module>
    fire.Fire()
    ~~~~~~~~~^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/fire/core.py", line 135, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/fire/core.py", line 468, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
                                ~~~~~~~~~~~~~~~~~~~^
        component,
        ^^^^^^^^^^
    ...<2 lines>...
        treatment='class' if is_class else 'routine',
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        target=component.__name__)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/fire/core.py", line 684, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "/media/smbshare/srs-9/prl_project/training/roi_train2_t1/run_losswt115_numcrop2/segresnet_0/scripts/train.py", line 23, in run
    run_segmenter(config_file=config_file, **override)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/smbshare/srs-9/prl_project/training/roi_train2_t1/run_losswt115_numcrop2/segresnet_0/scripts/segmenter.py", line 2204, in run_segmenter
    run_segmenter_worker(0, config_file, kwargs)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/smbshare/srs-9/prl_project/training/roi_train2_t1/run_losswt115_numcrop2/segresnet_0/scripts/segmenter.py", line 2175, in run_segmenter_worker
    segmenter = Segmenter(config_file=config_file, config_dict=override, rank=rank, global_rank=global_rank)
  File "/media/smbshare/srs-9/prl_project/training/roi_train2_t1/run_losswt115_numcrop2/segresnet_0/scripts/segmenter.py", line 663, in __init__
    loss_function = ConfigParser(config["loss"]).get_parsed_content(instantiate=True)
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/bundle/config_parser.py", line 290, in get_parsed_content
    return self.ref_resolver.get_resolved_content(id=id, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/bundle/reference_resolver.py", line 193, in get_resolved_content
    return self._resolve_one_item(id=id, **kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/bundle/reference_resolver.py", line 163, in _resolve_one_item
    self._resolve_one_item(id=d, waiting_list=waiting_list, **kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/bundle/reference_resolver.py", line 163, in _resolve_one_item
    self._resolve_one_item(id=d, waiting_list=waiting_list, **kwargs)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/bundle/reference_resolver.py", line 159, in _resolve_one_item
    raise ValueError(msg) from err
ValueError: the referring item `@sigmoid` is not defined in the config content.
Traceback (most recent call last):
  File "/home/srs-9/.virtualenvs/monai/bin/prl", line 8, in <module>
    sys.exit(cli())
             ~~~^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/click/core.py", line 1485, in __call__
    return self.main(*args, **kwargs)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/click/core.py", line 1406, in main
    rv = self.invoke(ctx)
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/click/core.py", line 1873, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
                           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/click/core.py", line 1269, in invoke
    return ctx.invoke(self.callback, **ctx.params)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/click/core.py", line 824, in invoke
    return callback(*args, **kwargs)
  File "/home/srs-9/Projects/prl_project/src/cli.py", line 127, in train
    exp.train()
    ~~~~~~~~~^^
  File "/home/srs-9/Projects/prl_project/src/core/experiment.py", line 306, in train
    runner.run()
    ~~~~~~~~~~^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/apps/auto3dseg/auto_runner.py", line 887, in run
    self._train_algo_in_sequence(history)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/apps/auto3dseg/auto_runner.py", line 737, in _train_algo_in_sequence
    algo.train(self.train_params, self.device_setting)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/apps/auto3dseg/bundle_gen.py", line 300, in train
    return self._run_cmd(cmd)
           ~~~~~~~~~~~~~^^^^^
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/apps/auto3dseg/bundle_gen.py", line 277, in _run_cmd
    return run_cmd(cmd.split(), run_cmd_verbose=True, env=ps_environ, check=True)
  File "/home/srs-9/.virtualenvs/monai/lib/python3.13/site-packages/monai/utils/misc.py", line 889, in run_cmd
    return subprocess.run(cmd_list, **kwargs)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/srs-9/.pyenv/versions/3.13.0/lib/python3.13/subprocess.py", line 577, in run
    raise CalledProcessError(retcode, process.args,
                             output=stdout, stderr=stderr)
subprocess.CalledProcessError: Command '['python', '/media/smbshare/srs-9/prl_project/training/roi_train2_t1/run_losswt115_numcrop2/segresnet_0/scripts/train.py', 'run', "--config_file='/media/smbshare/srs-9/prl_project/training/roi_train2_t1/run_losswt115_numcrop2/segresnet_0/configs/hyper_parameters.yaml'"]' returned non-zero exit status 1.


9: batch_size: 4
104: num_crops_per_image: 1
271: Given num_crops_per_image 4, num_epochs was adjusted 500 => 166
276:  batch_size => 1 
277:  num_crops_per_image => 4 



got it. next im trying to understand why segresnet adjusts the epochs down when you set num_crops_per_image higher than 1. for instance: "Given num_crops_per_image 4, num_epochs was adjusted 500 => 166". 

Lets take "~/hpc/prl_project/training/roi_train2/stage3_numcrops_bkd_constwt115" as an example. In run4, num_crops_per_image=4. You'll see in "~/hpc/prl_project/training/roi_train2/stage3_numcrops_bkd_constwt115/logs/run_162720_4.err" that each epoch takes around 20-40 seconds. Then take run1 as an example where num_crops_per_image=1 so epochs stay at 500. Yet each epoch still takes around the same time (see ~/hpc/prl_project/training/roi_train2/stage3_numcrops_bkd_constwt115/logs/run_162720_1.err, same time per epoch).

So to me it seems like the same amount of "work" is being done per epoch, but less total work is being done with num_crops_per_image=4 (it finishes almost 4 times faster), so will I even be able to compare these side by side? Should I have checked to prevent it from adjusting the epochs down? Or do i have some fundamental misunderstanding.


got it. so I will also submit the three runs that will be produced by `~/hpc/prl_project/prl_project/training/roi_train2/experiment_config_stage5.json`. So that tomorrow I have everything in front of me (i need to finalize an abstract by Friday). 



ok i have updated information for you. I see your previous responses were heavily informed by old info about my labels and results, like you kept coming back to that dice and the label quality.

Check out these results of hyperparameter sweeps on the new data in descending order of prl_dice (the performance from the model trained on the old data in the last row). 

In the ID column, roi_train2 means new data. Everything was trained on 2 channel flair.phase image stacks unless ID has roi_train2_t1, in which case T1 was an additional 3rd channel.

To calculate TP, FP, TN, FN: rim voxels (label index 2) were considered for TP, and lesion voxels (label index 1) were considered TN. Essentially:

```python
TP = np.sum((lab_data == 2) & (inf_data == 2))
FP = np.sum((lab_data == 1) & (inf_data == 2))
TN = np.sum((lab_data == 1) & (inf_data == 1))
FN = np.sum((lab_data == 2) & (inf_data == 1))
```

I just aggregated these over all the cases and computed values like sensitivity and precision from the cumulative counts. But the last several columns do show mean and std of metrics derived on a case by case basis.

First look carefully at what parameters I varied since some important ones were only changed in one or two runs. Columns 0-9 are the parameters. The next few columns are statistics about the number of cases and voxels, and then metrics.