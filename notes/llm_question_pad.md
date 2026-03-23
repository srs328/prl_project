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