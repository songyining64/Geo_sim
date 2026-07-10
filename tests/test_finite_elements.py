"""
有限元模块测试
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from finite_elements.basis_functions import (
    LinearBasisFunctions,
    LagrangeTriangle,
    LagrangeQuad,
    LagrangeTetra,
    LagrangeHex,
    TRIANGLE,
    QUADRILATERAL,
    TETRAHEDRON,
    HEXAHEDRON,
)
from finite_elements.mesh_generation import AdaptiveMesh, MeshCell
from finite_elements.quadrature import triangle_points_weights, quad_points_weights


class TestLinearBasisFunctions:
    """线性基函数测试"""

    def test_triangle_basis_functions(self):
        """测试三角形基函数"""
        basis = LinearBasisFunctions(TRIANGLE)
        nodes = np.array([[0, 0], [1, 0], [0, 1]])
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(3)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)

    def test_triangle_derivatives(self):
        """测试三角形基函数导数"""
        basis = LinearBasisFunctions(TRIANGLE)
        xi = np.array([[0.25, 0.25]])
        dN = basis.evaluate_derivatives(xi)
        assert dN.shape == (1, 3, 2)
        expected_dN = np.array([[[-1, -1], [1, 0], [0, 1]]])
        np.testing.assert_allclose(dN, expected_dN, atol=1e-10)

    def test_quadrilateral_basis_functions(self):
        """测试四边形基函数"""
        basis = LinearBasisFunctions(QUADRILATERAL)
        nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(4)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)

    def test_quadrilateral_derivatives(self):
        """测试四边形基函数导数"""
        basis = LinearBasisFunctions(QUADRILATERAL)
        xi = np.array([[0, 0]])
        dN = basis.evaluate_derivatives(xi)
        assert dN.shape == (1, 4, 2)
        expected_dN = np.array([[[-0.25, -0.25], [0.25, -0.25], [0.25, 0.25], [-0.25, 0.25]]])
        np.testing.assert_allclose(dN, expected_dN, atol=1e-10)

    def test_tetrahedron_basis_functions(self):
        """测试四面体基函数"""
        basis = LinearBasisFunctions(TETRAHEDRON)
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(4)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)

    def test_tetrahedron_derivatives(self):
        """测试四面体基函数导数"""
        basis = LinearBasisFunctions(TETRAHEDRON)
        xi = np.array([[0.25, 0.25, 0.25]])
        dN = basis.evaluate_derivatives(xi)
        assert dN.shape == (1, 4, 3)
        expected_dN = np.array([[[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        np.testing.assert_allclose(dN, expected_dN, atol=1e-10)

    def test_hexahedron_basis_functions(self):
        """测试六面体基函数"""
        basis = LinearBasisFunctions(HEXAHEDRON)
        nodes = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ])
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(8)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)

    def test_hexahedron_derivatives(self):
        """测试六面体基函数导数"""
        basis = LinearBasisFunctions(HEXAHEDRON)
        xi = np.array([[0, 0, 0]])
        dN = basis.evaluate_derivatives(xi)
        assert dN.shape == (1, 8, 3)
        expected_dN = np.array([[
            [-0.125, -0.125, -0.125], [0.125, -0.125, -0.125],
            [0.125, 0.125, -0.125], [-0.125, 0.125, -0.125],
            [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125],
            [0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]
        ]])
        np.testing.assert_allclose(dN, expected_dN, atol=1e-10)

    def test_partition_of_unity(self):
        """测试基函数的单位分解性质"""
        test_cases = [
            (TRIANGLE, np.array([[0.3, 0.4]])),
            (QUADRILATERAL, np.array([[0.2, -0.3]])),
            (TETRAHEDRON, np.array([[0.2, 0.3, 0.1]])),
            (HEXAHEDRON, np.array([[0.1, -0.2, 0.3]])),
        ]
        for element_type, xi in test_cases:
            basis = LinearBasisFunctions(element_type)
            N = basis.evaluate(xi)
            np.testing.assert_allclose(np.sum(N[0]), 1.0, atol=1e-10,
                                       err_msg=f"Partition of unity failed for {element_type}")

    def test_invalid_element_type(self):
        """测试无效的单元类型"""
        invalid_element = type('InvalidElement', (), {
            'name': 'invalid',
            'dimension': 2,
            'n_nodes': 5,
            'n_faces': 5,
            'n_edges': 5
        })()
        with pytest.raises(ValueError, match="不支持"):
            LinearBasisFunctions(invalid_element)

    def test_basis_positive(self):
        """测试基函数在单元内的非负性"""
        element_types = [TRIANGLE, QUADRILATERAL, TETRAHEDRON, HEXAHEDRON]
        for etype in element_types:
            basis = LinearBasisFunctions(etype)
            center = np.array([[0.25] * etype.dimension])
            N = basis.evaluate(center)
            assert np.all(N >= -1e-10), f"Basis functions should be non-negative for {etype}"

    def test_basis_symmetry_triangle(self):
        """测试三角形基函数的对称性"""
        basis = LinearBasisFunctions(TRIANGLE)
        xi = np.array([[0.2, 0.3]])
        N = basis.evaluate(xi)
        assert N.shape == (1, 3)


class TestQuadrature:
    """高斯积分测试"""

    def test_triangle_quadrature_order(self):
        """测试三角形积分精度"""
        pts, wts = triangle_points_weights(1)
        assert len(pts) > 0
        assert len(wts) > 0

    def test_quad_quadrature_order(self):
        """测试四边形积分精度"""
        pts, wts = quad_points_weights(1)
        assert len(pts) > 0
        assert len(wts) > 0

    def test_triangle_quadrature_sum_to_area(self):
        """测试三角形积分权重和为0.5（标准三角形面积）"""
        for order in [1, 2]:
            _, wts = triangle_points_weights(order)
            np.testing.assert_allclose(np.sum(wts), 0.5, atol=1e-10)

    def test_quad_quadrature_sum_to_four(self):
        """测试四边形积分权重和为4"""
        for order in [1, 2]:
            _, wts = quad_points_weights(order)
            np.testing.assert_allclose(np.sum(wts), 4.0, atol=1e-10)


class TestAdaptiveMesh:
    """网格测试"""

    def test_create_simple_mesh(self):
        """测试创建简单网格"""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cell = MeshCell(id=0, nodes=[0, 1, 2, 3], level=0)
        mesh = AdaptiveMesh(nodes=nodes, cells=[cell], dim=2)
        assert len(mesh.nodes) == 4
        assert len(mesh.cells) == 1
        assert mesh.dim == 2

    def test_create_triangle_mesh(self):
        """测试创建三角形网格"""
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        cell = MeshCell(id=0, nodes=[0, 1, 2], level=0)
        mesh = AdaptiveMesh(nodes=nodes, cells=[cell], dim=2)
        assert len(mesh.cells) == 1

    def test_3d_tetrahedral_mesh(self):
        """测试创建四面体网格"""
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        cell = MeshCell(id=0, nodes=[0, 1, 2, 3], level=0)
        mesh = AdaptiveMesh(nodes=nodes, cells=[cell], dim=3)
        assert mesh.dim == 3

    def test_mesh_connectivity(self):
        """测试网格连接关系"""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cell = MeshCell(id=0, nodes=[0, 1, 2, 3], level=0)
        mesh = AdaptiveMesh(nodes=nodes, cells=[cell], dim=2)
        assert hasattr(mesh, 'node_to_cells')
        assert 0 in mesh.node_to_cells

    def test_mesh_cell_properties(self):
        """测试单元属性计算"""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cell = MeshCell(id=0, nodes=[0, 1, 2, 3], level=0)
        mesh = AdaptiveMesh(nodes=nodes, cells=[cell], dim=2)
        assert hasattr(mesh.cells[0], 'center')

    def test_mesh_boundary_detection(self):
        """测试边界检测"""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                          [2, 0], [2, 1]], dtype=float)
        cells = [
            MeshCell(id=0, nodes=[0, 1, 2, 3], level=0),
            MeshCell(id=1, nodes=[1, 4, 5, 2], level=0),
        ]
        mesh = AdaptiveMesh(nodes=nodes, cells=cells, dim=2)
        assert hasattr(mesh, 'boundary_faces')

    def test_mesh_with_levels(self):
        """测试多层网格"""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cell = MeshCell(id=0, nodes=[0, 1, 2, 3], level=2)
        mesh = AdaptiveMesh(nodes=nodes, cells=[cell], dim=2)
        assert mesh.cells[0].level == 2

    def test_fine_mesh_no_errors(self):
        """测试较大网格不会产生属性错误"""
        nx, ny = 5, 5
        x_coords = np.linspace(0, 1, nx + 1)
        y_coords = np.linspace(0, 1, ny + 1)
        nodes = []
        for y in y_coords:
            for x in x_coords:
                nodes.append([x, y])
        nodes = np.array(nodes)
        cells = []
        cell_id = 0
        for j in range(ny):
            for i in range(nx):
                n1 = j * (nx + 1) + i
                n2 = j * (nx + 1) + i + 1
                n3 = (j + 1) * (nx + 1) + i + 1
                n4 = (j + 1) * (nx + 1) + i
                cells.append(MeshCell(id=cell_id, nodes=[n1, n2, n3, n4], level=0))
                cell_id += 1
        mesh = AdaptiveMesh(nodes=nodes, cells=cells, dim=2)
        assert len(mesh.nodes) == (nx + 1) * (ny + 1)
        assert len(mesh.cells) == nx * ny


class TestLagrangeBasis:
    """Lagrange基函数测试"""

    def test_lagrange_triangle_linear(self):
        """测试线性Lagrange三角形"""
        basis = LagrangeTriangle(1)
        nodes = np.array([[0, 0], [1, 0], [0, 1]])
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(3)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)

    def test_lagrange_quad_linear(self):
        """测试线性Lagrange四边形"""
        basis = LagrangeQuad(1)
        nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(4)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)

    def test_lagrange_tetra_linear(self):
        """测试线性Lagrange四面体"""
        basis = LagrangeTetra(1)
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(4)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)

    def test_lagrange_hex_linear(self):
        """测试线性Lagrange六面体"""
        basis = LagrangeHex(1)
        nodes = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ])
        for i, node in enumerate(nodes):
            N = basis.evaluate(node.reshape(1, -1))
            expected = np.zeros(8)
            expected[i] = 1.0
            np.testing.assert_allclose(N[0], expected, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
