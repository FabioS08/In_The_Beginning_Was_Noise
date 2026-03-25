from configs.schema import Config, ModelConfig, DatasetConfig, TrainingConfig

config = Config(

                    model = ModelConfig(

                                            name = "ddpm",
                                            timesteps = 1000,
                                            varianceSchedule = "linear",
                                            varianceScheduleParams = {"betaStart": 1e-4, "betaEnd": 0.02},
                                            timeEmbeddingDimension = 128,
                                        ),

                    dataset = DatasetConfig(
                                                name = "celebahq",
                                                rootPath = "./datasets",
                                                download = False,
                                            ),

                    training = TrainingConfig(
                                                    batchSize = 32,
                                                    epochs = 500,
                                                    learningRate = 0.0001,
                                                    optimizer = 'AdamW',
                                                    loss = 'MSE',
                                                    gradientAccumulationSteps = 1,
                                                    ampType = 'None',
                                                    checkpointSavingFrequency = 5,
                                                    maxNumCheckpoints = 2,
                                                    checkpointPathRestart = '',
                                                    maxGradNorm = 10,
                                                    checkpointDirectory = ''

                                                )
)
