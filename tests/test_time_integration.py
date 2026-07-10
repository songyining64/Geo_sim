"""
时间积分器测试
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from time_integration.time_integrators import (
    TimeIntegrator,
    RungeKuttaIntegrator,
    AdaptiveTimeIntegrator,
    ImplicitTimeIntegrator,
)
from time_integration.advanced_integrators import (
    BDFIntegrator,
    CrankNicolsonIntegrator,
)


class TestRungeKuttaIntegrator:
    """Runge-Kutta积分器测试"""

    def test_rk1_forward_euler(self):
        """测试RK1（前向Euler）"""
        rk = RungeKuttaIntegrator(order=1)
        
        def system(t, y):
            return -y
        
        y0 = np.array([1.0])
        dt = 0.01
        y = rk.integrate(dt, system, y0)
        assert rk.steps_taken == 1
        assert y[0] < 1.0

    def test_rk4_exponential_decay(self):
        """测试RK4指数衰减"""
        rk = RungeKuttaIntegrator(order=4)
        
        def system(t, y):
            return -0.5 * y
        
        y0 = np.array([1.0])
        dt = 0.1
        n_steps = 10
        
        y = y0.copy()
        for _ in range(n_steps):
            y = rk.integrate(dt, system, y)
        
        expected = np.exp(-0.5 * dt * n_steps)
        np.testing.assert_allclose(y[0], expected, rtol=1e-6)

    def test_rk4_constant_system(self):
        """测试RK4常速率系统"""
        rk = RungeKuttaIntegrator(order=4)
        
        def system(t, y):
            return np.array([2.0])
        
        y0 = np.array([0.0])
        dt = 0.1
        y = rk.integrate(dt, system, y0)
        np.testing.assert_allclose(y[0], 0.2, rtol=1e-10)

    def test_rk4_oscillator(self):
        """测试RK4简谐振子"""
        rk = RungeKuttaIntegrator(order=4)
        
        def system(t, y):
            return np.array([y[1], -y[0]])
        
        y0 = np.array([1.0, 0.0])
        dt = 0.01
        y = rk.integrate(dt, system, y0)
        assert abs(y[0]**2 + y[1]**2 - 1.0) < 1e-4

    def test_rk_invalid_order(self):
        """测试无效阶数"""
        with pytest.raises(ValueError):
            RungeKuttaIntegrator(order=5)

    def test_rk_get_integration_info(self):
        """测试获取积分信息"""
        rk = RungeKuttaIntegrator(order=4)
        info = rk.get_integration_info()
        assert info['order'] == 4
        assert info['steps_taken'] == 0


class TestAdaptiveTimeIntegrator:
    """自适应时间步长测试"""

    def test_adaptive_creation(self):
        """测试创建自适应积分器"""
        ti = AdaptiveTimeIntegrator(tolerance=1e-6)
        assert ti.dt > 0

    def test_adaptive_constant_steps(self):
        """测试常速率系统"""
        ti = AdaptiveTimeIntegrator(tolerance=1e-6, min_dt=1e-8, max_dt=0.1)
        
        def system(t, y):
            return np.array([1.0])
        
        y = np.array([0.0])
        dt = 0.01
        
        y_new = ti.integrate(dt, system, y)
        assert len(y_new) == len(y)


class TestBDFIntegrator:
    """BDF积分器测试"""

    def test_bdf_creation(self):
        """测试创建BDF积分器"""
        bdf = BDFIntegrator(order=2)
        assert bdf.order == 2

    def test_bdf_order_range(self):
        """测试BDF阶数范围"""
        for order in [1, 2, 3, 4]:
            bdf = BDFIntegrator(order=order)
            assert bdf.order == order


class TestCrankNicolson:
    """Crank-Nicolson积分器测试"""

    def test_cn_creation(self):
        """测试创建CN积分器"""
        cn = CrankNicolsonIntegrator()
        assert cn is not None

    def test_cn_theta(self):
        """测试CN阶数"""
        cn = CrankNicolsonIntegrator(order=2)
        assert cn.order == 2


class TestImplicitTimeIntegrator:
    """隐式时间积分器测试"""

    def test_implicit_creation(self):
        """测试创建隐式积分器"""
        ti = ImplicitTimeIntegrator(order=2)
        assert ti is not None
        assert ti.order == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
