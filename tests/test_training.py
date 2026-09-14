# Third-party
import pytorch_lightning as pl

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.graphcast import GraphCast
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import DummyModelArgs, create_test_graph


def test_training(datastore):
    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        detect_anomaly=True,
    )

    graph_name = "1level"
    create_test_graph(datastore, name=graph_name)

    data_module = WeatherDataModule(
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        ar_steps_train=2,
        ar_steps_eval=2,
        input_steps=2,
        standardize=True,
        batch_size=1,
        num_workers=0,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )

    model_args = DummyModelArgs()
    model_args.graph = graph_name

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )

    model = GraphCast(
        args=model_args,
        datastore=datastore,
        datastore_boundary=None,
        datastore_atmosphere=None,
        config=config,
    )
    trainer.fit(model=model, datamodule=data_module)
