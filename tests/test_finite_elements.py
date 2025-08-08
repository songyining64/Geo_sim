"""
有限元模块测试
"""

import numpy as np
import pytest
from finite_elements.basis_functions import (
    LinearBasisFunctions,
    TRIANGLE,
    QUADRILATERAL,
    TETRAHEDRON,
    HEXAHEDRON,
)


class TestLinearBasisFunctions:
    """线性基函数测试"""
    
    def test_triangle_basis_functions(self):
        """测试三角形基函数"""
        basis = LinearBasisFunctions(TRIANGLE)
        
        # 测试节点处的基函数值
        nodes = np.array([[0, 0], [1, 0], [0, 1]])  # 三个节点
        
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(3)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)
    
    def test_triangle_derivatives(self):
        """测试三角形基函数导数"""
        basis = LinearBasisFunctions(TRIANGLE)
        
        # 测试任意点的导数
        xi = np.array([[0.25, 0.25]])
        dN = basis.evaluate_derivatives(xi)
        
        # 检查导数矩阵形状
        assert dN.shape == (1, 3, 2)
        
        # 检查导数值（三角形线性基函数的导数是常数）
        expected_dN = np.array([[[-1, -1], [1, 0], [0, 1]]])
        np.testing.assert_allclose(dN, expected_dN, atol=1e-10)
    
    def test_quadrilateral_basis_functions(self):
        """测试四边形基函数"""
        basis = LinearBasisFunctions(QUADRILATERAL)
        
        # 测试节点处的基函数值
        nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])  # 四个节点
        
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(4)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)
    
    def test_quadrilateral_derivatives(self):
        """测试四边形基函数导数"""
        basis = LinearBasisFunctions(QUADRILATERAL)
        
        # 测试中心点的导数
        xi = np.array([[0, 0]])
        dN = basis.evaluate_derivatives(xi)
        
        # 检查导数矩阵形状
        assert dN.shape == (1, 4, 2)
        
        # 检查中心点的导数值
        expected_dN = np.array([[[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]]])
        np.testing.assert_allclose(dN, expected_dN, atol=1e-10)
    
    def test_tetrahedron_basis_functions(self):
        """测试四面体基函数"""
        basis = LinearBasisFunctions(TETRAHEDRON)
        
        # 测试节点处的基函数值
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 四个节点
        
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(4)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)
    
    def test_tetrahedron_derivatives(self):
        """测试四面体基函数导数"""
        basis = LinearBasisFunctions(TETRAHEDRON)
        
        # 测试任意点的导数
        xi = np.array([[0.25, 0.25, 0.25]])
        dN = basis.evaluate_derivatives(xi)
        
        # 检查导数矩阵形状
        assert dN.shape == (1, 4, 3)
        
        # 检查导数值（四面体线性基函数的导数是常数）
        expected_dN = np.array([[[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        np.testing.assert_allclose(dN, expected_dN, atol=1e-10)
    
    def test_hexahedron_basis_functions(self):
        """测试六面体基函数"""
        basis = LinearBasisFunctions(HEXAHEDRON)
        
        # 测试节点处的基函数值
        nodes = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ])  # 八个节点
        
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(8)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)
    
    def test_hexahedron_derivatives(self):
        """测试六面体基函数导数"""
        basis = LinearBasisFunctions(HEXAHEDRON)
        
        # 测试中心点的导数
        xi = np.array([[0, 0, 0]])
        dN = basis.evaluate_derivatives(xi)
        
        # 检查导数矩阵形状
        assert dN.shape == (1, 8, 3)
        
        # 检查中心点的导数值
        expected_dN = np.array([[
            [-0.125, -0.125, -0.125], [0.125, -0.125, -0.125],
            [0.125, 0.125, -0.125], [-0.125, 0.125, -0.125],
            [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125],
            [0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]
        ]])
        np.testing.assert_allclose(dN, expected_dN, atol=1e-10)
    
    def test_partition_of_unity(self):
        """测试基函数的单位分解性质"""
        # 测试三角形
        basis = LinearBasisFunctions(TRIANGLE)
        xi = np.array([[0.3, 0.4]])
        N = basis.evaluate(xi)
        np.testing.assert_allclose(np.sum(N[0]), 1.0, atol=1e-10)
        
        # 测试四边形
        basis = LinearBasisFunctions(QUADRILATERAL)
        xi = np.array([[0.2, -0.3]])
        N = basis.evaluate(xi)
        np.testing.assert_allclose(np.sum(N[0]), 1.0, atol=1e-10)
        
        # 测试四面体
        basis = LinearBasisFunctions(TETRAHEDRON)
        xi = np.array([[0.2, 0.3, 0.1]])
        N = basis.evaluate(xi)
        np.testing.assert_allclose(np.sum(N[0]), 1.0, atol=1e-10)
        
        # 测试六面体
        basis = LinearBasisFunctions(HEXAHEDRON)
        xi = np.array([[0.1, -0.2, 0.3]])
        N = basis.evaluate(xi)
        np.testing.assert_allclose(np.sum(N[0]), 1.0, atol=1e-10)
    
    def test_invalid_element_type(self):
        """测试无效的单元类型"""
        invalid_element = type('InvalidElement', (), {
            'name': 'invalid',
            'dimension': 2,
            'n_nodes': 5,
            'n_faces': 5,
            'n_edges': 5
        })()
        
        with pytest.raises(ValueError, match="不支持的单元类型"):
            LinearBasisFunctions(invalid_element)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 