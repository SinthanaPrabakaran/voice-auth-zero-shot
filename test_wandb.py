import wandb

wandb.init(
    project="voice-auth-cnn",
    name="test-run",
    notes="Just checking if logging works properly"
)

wandb.log({
    "accuracy": 0.95,
    "loss": 0.02
})

wandb.finish()
