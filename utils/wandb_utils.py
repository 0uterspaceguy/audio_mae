import wandb

def init_wandb_project(exp_label, wandb_project, wandb_user):
    wandb.init(project=wandb_project, entity=wandb_user, name=exp_label, settings=wandb.Settings(code_dir="."))
    wandb.run.log_code(".")
    wandb.save('configs/base_config.yaml')
    wandb.save('configs/current_config.yaml')


def wandb_log_metrics(metrics):
    wandb.log(metrics)
