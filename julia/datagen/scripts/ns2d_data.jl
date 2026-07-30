using GeophysicalFlows


grid = RectilinearGrid(
    topology=(Periodic, Periodic),
    size=(256, 256),
    x = (0,1),
    y = (0,1))

model = NonhydrostaticModel(grid)


simulation = Simulation(model; Δt=1e-4, stop_iteration=50)
run!(simulation)
