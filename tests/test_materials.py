"""
材料模型测试
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from materials.advanced_material_models import (
    Material,
    MaterialRegistry,
    MaterialState,
    ConstantViscosity,
    PowerLawCreep,
    TemperatureDepthViscosity,
    AdvancedDensity,
    create_crust_material,
    create_mantle_material,
    create_air_material,
)
from materials.plastic_models import (
    VonMisesPlasticity,
    DruckerPragerPlasticity,
    PlasticState,
)


class TestMaterialBase:
    """Material基类测试"""

    def test_create_material(self):
        """测试创建基本材料"""
        mat = Material(name="test", density=2700.0)
        assert mat.name == "test"
        assert isinstance(mat.density, AdvancedDensity)

    def test_material_default_values(self):
        """测试材料默认值"""
        mat = Material(name="default")
        assert mat.thermal_conductivity == 3.0
        assert mat.heat_capacity == 1000.0
        assert mat.thermal_expansivity == 3e-5


class TestAdvancedDensity:
    """密度模型测试"""

    def test_constant_density(self):
        """测试常密度"""
        density = AdvancedDensity(reference_density=2700.0)
        t = np.array([300.0])
        p = np.array([1e5])
        result = density.compute_density(t, p)
        np.testing.assert_allclose(result, 2700.0, rtol=0.01)

    def test_density_temperature_dependence(self):
        """测试密度温度依赖性"""
        density = AdvancedDensity(
            reference_density=3300.0,
            thermal_expansivity=3e-5
        )
        t0 = density.reference_temperature
        t_hot = np.array([t0 + 1000.0])
        p = np.array([1e5])
        result = density.compute_density(t_hot, p)
        assert result[0] < 3300.0, "Density should decrease with temperature"

    def test_density_pressure_dependence(self):
        """测试密度压力依赖性"""
        density = AdvancedDensity(
            reference_density=3300.0,
            compressibility=1e-11
        )
        t = np.array([density.reference_temperature])
        p_high = np.array([1e9])
        result = density.compute_density(t, p_high)
        assert result[0] > 3300.0, "Density should increase with pressure"


class TestRheology:
    """流变学模型测试"""

    def test_constant_viscosity(self):
        """测试常粘度"""
        state = MaterialState(
            temperature=np.array([500.0]),
            pressure=np.array([1e8]),
            strain_rate=np.array([1e-15]),
            plastic_strain=np.array([0.0]),
        )
        rheology = ConstantViscosity(viscosity=1e21)
        rheology.set_material_state(state)
        mu = rheology.compute_effective_viscosity()
        np.testing.assert_allclose(mu, 1e21, rtol=1e-10)

    def test_power_law_creep_basic(self):
        """测试幂律蠕变"""
        state = MaterialState(
            temperature=np.array([1500.0]),
            pressure=np.array([1e9]),
            strain_rate=np.array([[1e-15]]),  # 2D for invariant calc
            plastic_strain=np.array([0.0]),
        )
        rheology = PowerLawCreep(
            pre_exponential_factor=1e-16,
            stress_exponent=3.5,
            activation_energy=500e3,
            activation_volume=10e-6,
            name="Test Power Law",
        )
        rheology.set_material_state(state)
        mu = rheology.compute_effective_viscosity()
        assert mu > 0
        assert np.all(np.isfinite(mu))

    def test_temperature_depth_viscosity(self):
        """测试温度-深度粘度"""
        state = MaterialState(
            temperature=np.array([500.0, 1500.0]),
            pressure=np.array([-5e7, -1e9]),  # negative for depth proxy
            strain_rate=np.array([[1e-15], [1e-15]]),
            plastic_strain=np.array([0.0, 0.0]),
        )
        rheology = TemperatureDepthViscosity(
            reference_viscosity=1e21,
            temperature_factor=-0.001,
            depth_factor=0.0,
            reference_depth=0.0,
        )
        rheology.set_material_state(state)
        mu = rheology.compute_effective_viscosity()
        assert mu.shape == (2,)
        assert np.all(mu > 0)


class TestMaterialConvenienceFunctions:
    """便捷材料创建函数测试"""

    def test_create_crust_material(self):
        """测试创建地壳材料"""
        mat = create_crust_material()
        assert "Crust" in mat.name
        assert mat.thermal_conductivity > 0

    def test_create_mantle_material(self):
        """测试创建地幔材料"""
        mat = create_mantle_material()
        assert "Mantle" in mat.name
        assert mat.thermal_expansivity > 0

    def test_create_air_material(self):
        """测试创建空气材料"""
        mat = create_air_material()
        assert mat.name == "Air"
        assert mat.density.reference_density < 10.0


class TestPlasticity:
    """塑性模型测试"""

    def test_von_mises_yield(self):
        """测试Von Mises屈服"""
        plasticity = VonMisesPlasticity(yield_stress=1e6)
        assert plasticity.yield_stress == 1e6

    def test_drucker_prager_yield(self):
        """测试Drucker-Prager屈服"""
        plasticity = DruckerPragerPlasticity(
            cohesion=1e6,
            friction_angle=30.0,
        )
        assert plasticity.cohesion == 1e6


class TestMaterialState:
    """材料状态测试"""

    def test_empty_state(self):
        """测试空状态"""
        state = MaterialState(
            temperature=np.array([300.0]),
            pressure=np.array([1e5]),
            strain_rate=np.array([0.0]),
            plastic_strain=np.array([0.0]),
        )
        assert state.temperature is not None
        assert state.pressure is not None

    def test_state_with_arrays(self):
        """测试带数组的状态"""
        state = MaterialState(
            temperature=np.array([300.0, 500.0]),
            pressure=np.array([1e5, 1e6]),
            strain_rate=np.array([1e-15, 1e-14]),
            plastic_strain=np.array([0.0, 0.0]),
        )
        assert len(state.temperature) == 2


class TestMaterialRegistry:
    """材料注册系统测试"""

    def test_create_registry(self):
        """测试创建注册表"""
        registry = MaterialRegistry()
        assert len(registry.materials) == 3  # pre-loaded: crust, mantle, air

    def test_add_material(self):
        """测试添加材料"""
        registry = MaterialRegistry()
        mat = Material(name="granite", density=2700.0)
        registry.add_material("granite", mat)
        assert "granite" in registry.materials

    def test_get_material(self):
        """测试获取材料"""
        registry = MaterialRegistry()
        mat = Material(name="basalt", density=2900.0)
        registry.add_material("basalt", mat)
        retrieved = registry.get_material("basalt")
        assert retrieved.name == "basalt"

    def test_get_nonexistent(self):
        """测试获取不存在材料"""
        registry = MaterialRegistry()
        with pytest.raises(KeyError):
            registry.get_material("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
