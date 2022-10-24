from qstack.backends import IBMQQPU
import pytest
from qiskit import IBMQ


@pytest.fixture
def ibmq_provider():
    IBMQ.load_account()
    return IBMQ.get_provider(hub="ibm-q")


@pytest.mark.parametrize(
    "backend_name",
    ["FakeAthens", "FakeAthensV2", "FakeWashingtonV2"],
)
def test_passing_fake_backend_name_returns_expected_class_instance(backend_name):
    backend_instance = IBMQQPU(backend_name=backend_name)
    assert (
        backend_name.lower().replace("fake", "").replace("v2", "")
        in backend_instance._backend.backend_name.lower()
    )

    expected_version = 2 if "V2" in backend_name else 1
    assert backend_instance._backend.version == expected_version


@pytest.mark.parametrize(
    "backend_name",
    ["ibm_nairobi", "ibmq_manila"],
)
def test_passing_real_backend_name_returns_expected_class_instance(
    backend_name, ibmq_provider
):
    backend_instance = IBMQQPU(backend_name=backend_name, provider=ibmq_provider)
    assert backend_name.lower() in backend_instance._backend.name().lower()


def test_passing_fake_backend_and_provider_raises_warning(ibmq_provider):
    with pytest.warns(UserWarning):
        IBMQQPU(backend_name="FakeAthens", provider=ibmq_provider)


def test_nonfake_name_and_no_provider_raises_error():
    with pytest.raises(RuntimeError):
        IBMQQPU(backend_name="InvalidBackend")
