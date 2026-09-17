# Third-party
import pytorch_lightning as pl

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.graph_fm import GraphFM
from neural_lam.models.graphcast import GraphCast
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import DummyModelArgs, create_test_graph


def _cpu_trainer():
    return pl.Trainer(
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


def test_training(datastore):
    trainer = _cpu_trainer()

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


def test_training_graph_fm_with_boundary_and_atmosphere(
    datastore, datastore_boundary, datastore_atmosphere
):
    """Graph-FM train step with lateral g2m boundary and concat atmosphere."""
    trainer = _cpu_trainer()
    graph_name = "hierarchical"
    create_test_graph(
        datastore,
        name=graph_name,
        graph_type="hierarchical",
        n_max_levels=2,
        datastore_boundary=datastore_boundary,
    )

    data_module = WeatherDataModule(
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        ar_steps_train=2,
        ar_steps_eval=2,
        input_steps=2,
        standardize=True,
        batch_size=1,
        num_workers=0,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
        num_past_boundary_steps=1,
        num_future_boundary_steps=0,
        current_boundary_step=True,
        num_past_atmosphere_steps=1,
        num_future_atmosphere_steps=0,
        current_atmosphere_step=True,
        use_atmosphere_g2m=False,
    )

    model_args = DummyModelArgs()
    model_args.graph = graph_name
    model_args.use_atmosphere_g2m = False

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=str(datastore.root_path)
        )
    )

    model = GraphFM(
        args=model_args,
        datastore=datastore,
        datastore_boundary=datastore_boundary,
        datastore_atmosphere=datastore_atmosphere,
        config=config,
    )
    assert model.hierarchical
    assert model.boundary_forced
    assert model.concat_atmosphere
    trainer.fit(model=model, datamodule=data_module)
