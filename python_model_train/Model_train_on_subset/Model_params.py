from useful_classes import Model

model = Model()

# Count all parameters
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

# Count only trainable parameters
trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")