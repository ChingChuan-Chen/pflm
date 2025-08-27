import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.smooth import Polyfit1DModel
from pflm.smooth import KernelType


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit1d_happy_case(dtype):
    bw = 0.25
    x = np.linspace(0.0, 1.0, 11, dtype=dtype)
    y = 2.0 * x**2 + 3 * x
    w = np.ones_like(x)
    x_new = np.linspace(0.0, 1.0, 11, dtype=dtype)

    # fmt: off
    expected_results = {
        KernelType.GAUSSIAN: np.array([
            -0.059162939216, 0.325036778510, 0.731014910930, 1.161759855632,
            1.619430366873, 2.105204388503, 2.619430366873, 3.161759855632,
            3.731014910930, 4.325036778510, 4.940837060784
        ], dtype=dtype),
        KernelType.LOGISTIC: np.array([
            -0.161150328985, 0.277047311686, 0.724601740286, 1.184564011983,
            1.659446412861, 2.150885243892, 2.659446412861, 3.184564011983,
            3.724601740286, 4.277047311686, 4.838849671015
        ], dtype=dtype),
        KernelType.SIGMOID: np.array([
            -0.130518482307, 0.294439062982, 0.727032287138, 1.173968172887,
            1.640231007870, 2.128680341675, 2.640231007870, 3.173968172887,
            3.727032287138, 4.294439062982, 4.869481517693
        ], dtype=dtype),
        KernelType.RECTANGULAR: np.array([
            -0.006666666667, 0.340000000000, 0.720000000000, 1.120000000000,
            1.560000000000, 2.040000000000, 2.560000000000, 3.120000000000,
            3.720000000000, 4.340000000000, 4.993333333333
        ], dtype=dtype),
        KernelType.TRIANGULAR: np.array([
            -0.003157894737, 0.334482758621, 0.701538461538, 1.101538461538,
            1.541538461538, 2.021538461538, 2.541538461538, 3.101538461538,
            3.701538461538, 4.334482758621, 4.996842105263
        ], dtype=dtype),
        KernelType.EPANECHNIKOV: np.array([
            -0.004684014870, 0.337087794433, 0.706823529412, 1.106823529412,
            1.546823529412, 2.026823529412, 2.546823529412, 3.106823529412,
            3.706823529412, 4.337087794433, 4.995315985130
        ], dtype=dtype),
        KernelType.BIWEIGHT: np.array([
            -0.002780677479, 0.334288436982, 0.698334331935, 1.098334331935,
            1.538334331935, 2.018334331935, 2.538334331935, 3.098334331935,
            3.698334331935, 4.334288436982, 4.997219322521
        ], dtype=dtype),
        KernelType.TRIWEIGHT: np.array([
            -0.001370698495, 0.332100791473, 0.693680101109, 1.093680101109,
            1.533680101109, 2.013680101109, 2.533680101109, 3.093680101109,
            3.693680101109, 4.332100791473, 4.998629301505
        ], dtype=dtype),
        KernelType.TRICUBE: np.array([
            -0.002761917832, 0.334665254777, 0.697892313673, 1.097892313673,
            1.537892313673, 2.017892313673, 2.537892313673, 3.097892313673,
            3.697892313673, 4.334665254777, 4.997238082168
        ], dtype=dtype),
        KernelType.COSINE: np.array([
            -0.004357137153, 0.336602913011, 0.705278640450, 1.105278640450,
            1.545278640450, 2.025278640450, 2.545278640450, 3.105278640450,
            3.705278640450, 4.336602913011, 4.995642862847
        ], dtype=dtype),
    }
    # fmt: on

    for kernel_type, expected in expected_results.items():
        model = Polyfit1DModel(kernel_type=kernel_type)
        model.fit(x, y, sample_weight=w, bandwidth=bw, reg_grid=x_new)
        y_pred = model.predict(x_new)
        assert_allclose(y_pred, expected, rtol=1e-5, atol=1e-6, err_msg=f"Failed for kernel {kernel_type} with dtype {dtype}")


@pytest.mark.parametrize("order", ["C", "F"])
def test_polyfit1d_different_order_array(order):
    if order == "C":
        x = np.ascontiguousarray(np.linspace(0.0, 1.0, 11))
    else:
        x = np.asfortranarray(np.linspace(0.0, 1.0, 11))

    y = 2.0 * x**2 + 3 * x
    w = np.ones_like(x, order=order, dtype=x.dtype)
    if order == "C":
        x_new = np.ascontiguousarray(np.linspace(0.0, 1.0, 11))
    else:
        x_new = np.asfortranarray(np.linspace(0.0, 1.0, 11))

    # fmt: off
    expected_results = np.array([
        -0.059162939216, 0.325036778510, 0.731014910930, 1.161759855632,
        1.619430366873, 2.105204388503, 2.619430366873, 3.161759855632,
        3.731014910930, 4.325036778510, 4.940837060784
    ], order=order, dtype=x.dtype)
    # fmt: on

    model = Polyfit1DModel(kernel_type=KernelType.GAUSSIAN)
    model.fit(x, y, sample_weight=w, bandwidth=0.25, reg_grid=x_new)
    y_pred = model.predict(x_new)
    assert y_pred.shape == (len(x_new),)
    assert np.all(np.isfinite(y_pred)), "Prediction contains NaN or Inf values"
    assert_allclose(
        y_pred, expected_results, rtol=1e-5, atol=1e-6, err_msg=f"Failed for kernel {KernelType.GAUSSIAN} with dtype {x.dtype} and order {order}"
    )
    fitted_value = model.fitted_values()
    assert fitted_value.shape == (len(x),)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit1d_big(dtype):
    bw = 5.445
    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 6.0, 10.0, 8.0, 9.0], dtype=dtype)
    y = np.linspace(0.1, 1.0, 10, dtype=dtype)
    w = np.ones_like(x, dtype=dtype)

    x_new = np.linspace(1.0, 10.0, 21, dtype=dtype)
    sorted_idx = np.argsort(x)
    xs = x[sorted_idx]
    ys = y[sorted_idx]
    ws = w[sorted_idx]

    # fmt: off
    expected_results = {
        (KernelType.GAUSSIAN, 1, 0): np.array([
            0.114586353967, 0.159087772528, 0.203476488507, 0.247741547666,
            0.291870963774, 0.335851693572, 0.379669608445, 0.423309462831,
            0.466754859524, 0.509988212162, 0.552990705324, 0.595742252773,
            0.638221454523, 0.680405553527, 0.722270392867, 0.763790374458,
            0.804938420320, 0.845685937585, 0.886002788393, 0.925857265912,
            0.965216077679
        ], dtype=dtype),
        (KernelType.GAUSSIAN, 3, 1): np.array([
            0.066008431547, 0.074644211399, 0.083340851498, 0.091848185525,
            0.099909859941, 0.107263002444, 0.113637855948, 0.118757379937,
            0.122336821122, 0.124083255320, 0.123695102554, 0.120861617369,
            0.115262356409, 0.106566625290, 0.094432906872, 0.078508272999,
            0.058427781819, 0.033813862791, 0.004275691486, -0.030591443704,
            -0.071206780839
        ], dtype=dtype),
        (KernelType.LOGISTIC, 1, 0): np.array([
            0.120222650606, 0.163744104849, 0.207198694246, 0.250579480300,
            0.293879147177, 0.337090038203, 0.380204207606, 0.423213488598,
            0.466109578285, 0.508884139084, 0.551528915463, 0.594035863853,
            0.636397292677, 0.678606008587, 0.720655464275, 0.762539902773,
            0.804254492873, 0.845795450406, 0.887160140438, 0.928347156115,
            0.969356370812
        ], dtype=dtype),
        (KernelType.LOGISTIC, 3, 1): np.array([
            0.047653739505, 0.062763013359, 0.076516307075, 0.088820556267,
            0.099572292135, 0.108657906240, 0.115954088956, 0.121328443314,
            0.124640271990, 0.125741531166, 0.124477940845, 0.120690237049,
            0.114215547083, 0.104888864930, 0.092544599833, 0.077018167527,
            0.058147590613, 0.035775072501, 0.009748508441, -0.020077102260,
            -0.053838376686
        ], dtype=dtype),
        (KernelType.SIGMOID, 1, 0): np.array([
            0.117535400789, 0.161376166822, 0.205175309934, 0.248924320544,
            0.292613214023, 0.336229972379, 0.379759957990, 0.423185407269,
            0.466485124188, 0.509634482161, 0.552605810613, 0.595369198442,
            0.637893700107, 0.680148887311, 0.722106650574, 0.763743117971,
            0.805040522972, 0.845988826274, 0.886586890384, 0.926843033215,
            0.966774853216
        ], dtype=dtype),
        (KernelType.SIGMOID, 3, 1): np.array([
            0.057782941217, 0.069730188707, 0.080767699437, 0.090846692743,
            0.099887899826, 0.107779557398, 0.114376871942, 0.119502913112,
            0.122950715559, 0.124486286307, 0.123852263290, 0.120772129546,
            0.114955082781, 0.106101791453, 0.093911252319, 0.078088773510,
            0.058354787706, 0.034453856099, 0.006162977115, -0.026701744137,
            -0.064280850759
        ], dtype=dtype),
        (KernelType.RECTANGULAR, 1, 0): np.array([
            0.095238095238, 0.144095238095, 0.205000000000, 0.246785714286,
            0.286428571429, 0.333035714286, 0.381388888889, 0.430138888889,
            0.465454545455, 0.507727272727, 0.550000000000, 0.592272727273,
            0.634545454545, 0.677916666667, 0.719166666667, 0.760119047619,
            0.799761904762, 0.844821428571, 0.888214285714, 0.908142857143,
            0.942857142857
        ], dtype=dtype),
        (KernelType.RECTANGULAR, 3, 1): np.array([
            0.228835978836, 0.162934193122, 0.095551587302, 0.105328373016,
            0.092134199134, 0.085761634199, 0.089987373737, 0.089731691919,
            0.126868686869, 0.127353826729, 0.125242812743, 0.120535644911,
            0.113232323232, 0.119874188312, 0.104496753247, 0.110082972583,
            0.072544011544, 0.023156746032, -0.023182539683, -0.136672619048,
            -0.258333333333
        ], dtype=dtype),
        (KernelType.TRIANGULAR, 1, 0): np.array([
            0.109949225918, 0.153743661968, 0.200425885998, 0.246268779525,
            0.291103017639, 0.334323030918, 0.376867836726, 0.422017725182,
            0.469878907339, 0.515026357946, 0.560186757216, 0.605102364700,
            0.647267381681, 0.688441662153, 0.731396913225, 0.774743306363,
            0.815308401128, 0.851473658933, 0.881547049091, 0.897209516895,
            0.907550429225
        ], dtype=dtype),
        (KernelType.TRIANGULAR, 3, 1): np.array([
            0.215612371249, 0.159278528932, 0.110802207787, 0.101092980854,
            0.093596127066, 0.087469241029, 0.087501396280, 0.090925422490,
            0.097644252280, 0.113528316737, 0.119606182650, 0.122626032417,
            0.127979340998, 0.126831520357, 0.124062157970, 0.111581482574,
            0.078532601026, 0.032409268387, -0.038837704611, -0.176943052829,
            -0.364122194033
        ], dtype=dtype),
        (KernelType.EPANECHNIKOV, 1, 0): np.array([
            0.111069479239, 0.152700799416, 0.199552902871, 0.244701570772,
            0.288410270577, 0.332966817520, 0.378635913259, 0.425844450664,
            0.473212365964, 0.516403786619, 0.559346656993, 0.602050771672,
            0.644521566603, 0.686133188092, 0.727155392828, 0.767187169989,
            0.806377826185, 0.843945256986, 0.876817386601, 0.901555255376,
            0.925257248143
        ], dtype=dtype),
        (KernelType.EPANECHNIKOV, 3, 1): np.array([
            0.229396750139, 0.164413835612, 0.106915594550, 0.101274137788,
            0.095945565125, 0.090344514186, 0.088389016349, 0.089172508448,
            0.095784963718, 0.113962292181, 0.120314624390, 0.123715107751,
            0.130727954314, 0.128913098253, 0.123115157210, 0.108450169453,
            0.073252337907, 0.028718130837, -0.032377921384, -0.163308364042,
            -0.339602117920
        ], dtype=dtype),
        (KernelType.BIWEIGHT, 1, 0): np.array([
            0.115002887296, 0.157070554663, 0.200397387109, 0.244719374045,
            0.288803798957, 0.332885288999, 0.377481412777, 0.423125519980,
            0.469837762156, 0.516213134251, 0.561434287132, 0.605750136065,
            0.649301372036, 0.692022926066, 0.733724377854, 0.773923106409,
            0.811829372417, 0.845811328971, 0.874139006550, 0.896592667170,
            0.911781546260
        ], dtype=dtype),
        (KernelType.BIWEIGHT, 3, 1): np.array([
            0.214404272682, 0.163061635479, 0.113004592424, 0.095249223874,
            0.092400978870, 0.090969868595, 0.089872218543, 0.090548586930,
            0.093002848846, 0.104727096712, 0.116184842517, 0.125804933490,
            0.134168561613, 0.134211693244, 0.129743279955, 0.110215539750,
            0.081471265493, 0.037844925751, -0.038651929526, -0.186831047990,
            -0.418825443215,
        ], dtype=dtype),
        (KernelType.TRIWEIGHT, 1, 0): np.array([
            0.113001114942, 0.158347054201, 0.201875140491, 0.245428554724,
            0.289133568341, 0.332871422471, 0.376887220403, 0.421544968989,
            0.467217591174, 0.513775151351, 0.560403719915, 0.606560732993,
            0.651996454949, 0.696436199142, 0.739396872140, 0.780036547324,
            0.817089521692, 0.849072268501, 0.874737989373, 0.892821667321,
            0.900170361889,
        ], dtype=dtype),
        (KernelType.TRIWEIGHT, 3, 1): np.array([
            0.180504573483, 0.158560892349, 0.114852330664, 0.088620876181,
            0.082646356514, 0.086125052727, 0.089841255546, 0.093047607530,
            0.096267395721, 0.102642951162, 0.113472815155, 0.124557259004,
            0.132003883765, 0.134121267850, 0.129703113621, 0.116699404672,
            0.095498658622, 0.050626709052, -0.039773080582, -0.205910327509,
            -0.470485571637
        ], dtype=dtype),
        (KernelType.TRICUBE, 1, 0): np.array([
            0.118363401843, 0.158762257829, 0.199771562010, 0.243321937291,
            0.287955607484, 0.332716048590, 0.377819358749, 0.423598980559,
            0.470379846068, 0.517464314195, 0.563088672918, 0.606916842973,
            0.649412508465, 0.690901689713, 0.731412131864, 0.770706643877,
            0.807907166399, 0.841554046993, 0.871085754202, 0.897045388613,
            0.916354685006,
        ], dtype=dtype),
        (KernelType.TRICUBE, 3, 1): np.array([
            0.219422521870, 0.166796103059, 0.112794236615, 0.091461760435,
            0.092577527050, 0.094913381629, 0.092661070237, 0.090060510926,
            0.090505790673, 0.098986038715, 0.114176343004, 0.128972027254,
            0.138772847197, 0.139859972003, 0.130456060562, 0.107552861617,
            0.080026206501, 0.039771325160, -0.036880177625, -0.189356199086,
            -0.458450076277,
        ], dtype=dtype),
        (KernelType.COSINE, 1, 0): np.array([
            0.111720970469, 0.153515068467, 0.199741687065, 0.244715775958,
            0.288504167010, 0.332988833953, 0.378465124883, 0.425380293989,
            0.472584377196, 0.516316591777, 0.559704711972, 0.602765664685,
            0.645488660595, 0.687362556562, 0.728536926277, 0.768587792658,
            0.807486919800, 0.844348452719, 0.876439450438, 0.900773955625,
            0.923169695941,
        ], dtype=dtype),
        (KernelType.COSINE, 3, 1): np.array([
            0.227626520700, 0.164227973706, 0.108199157019, 0.100662451738,
            0.095521469870, 0.090408479833, 0.088601285009, 0.089412259329,
            0.095458633155, 0.112840199507, 0.119702679140, 0.123797153521,
            0.131045826247, 0.129455039268, 0.123799977245, 0.108671296838,
            0.074213949107, 0.029824216402, -0.033711867975, -0.167377210201,
            -0.349550294468,
        ], dtype=dtype),
    }
    # fmt: on

    for (kernel_type, degree, deriv), expected in expected_results.items():
        model = Polyfit1DModel(kernel_type=kernel_type, degree=degree, deriv=deriv)
        model.fit(xs, ys, sample_weight=ws, bandwidth=bw, reg_grid=x_new)
        y_pred = model.predict(x_new)
        assert_allclose(
            y_pred,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Failed for kernel {kernel_type} with dtype {dtype} and degree {degree} and deriv {deriv}",
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit1d_predict_random_order(dtype):
    bw = 1.35789
    x = np.array([1.0, 2.0, 3.0, 4.5, 5.0, 7.0], dtype=dtype)
    y = np.linspace(0.0, 1.0, 6, dtype=dtype)
    w = np.ones_like(x, dtype=dtype)
    x_new = np.array([3.0, 4.0, 6.0, 5.0, 5.5, 6.5], dtype=dtype)

    expected_results = np.array(
        [0.381629932858, 0.561082911319, 0.875183582904, 0.730103450143, 0.805520165855, 0.941180911169],
        dtype=dtype,
    )
    model = Polyfit1DModel(kernel_type=KernelType.GAUSSIAN)
    model.fit(x, y, sample_weight=w, bandwidth=bw, reg_grid=x_new)
    y_pred = model.predict(x_new)
    assert_allclose(y_pred, expected_results, rtol=1e-5, atol=1e-6, err_msg=f"Failed for dtype {dtype} with random order")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit1d_predict_nonunit_weight(dtype):
    bw = 1.35789
    x = np.array([1.0, 2.0, 3.0, 4.5, 5.0, 7.0], dtype=dtype)
    y = np.linspace(0.0, 1.0, 6, dtype=dtype)
    w = np.array([1.0, 1.0, 3.0, 4.0, 1.5, 2.5], dtype=dtype)
    x_new = np.array([3.0, 4.0, 6.0, 5.0, 5.5, 6.5], dtype=dtype)

    expected_results = np.array(
        [0.380560427220, 0.548544553298, 0.861230453851, 0.710492023351, 0.787257842890, 0.933244651929],
        dtype=dtype,
    )
    model = Polyfit1DModel(kernel_type=KernelType.GAUSSIAN)
    model.fit(x, y, sample_weight=w, bandwidth=bw, reg_grid=x_new)
    y_pred = model.predict(x_new)
    assert_allclose(y_pred, expected_results, rtol=1e-5, atol=1e-6, err_msg=f"Failed for dtype {dtype} with random order")


def make_test_inputs():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    x_new = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    return x, y, w, x_new


def test_polyfit1d_x_not_1d():
    x, y, w, x_new = make_test_inputs()

    # valid 2D input
    model = Polyfit1DModel()
    model.fit(x.reshape(-1, 1), y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)
    result = model.predict(x_new)
    assert result.shape == (len(x_new),)

    x = np.array([[0.1, 0.2], [0.3, 0.4]])  # Not 1D
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="X must have exactly 1 feature"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)


def test_polyfit1d_y_not_1d():
    x, y, w, x_new = make_test_inputs()
    y = np.array([[0.1, 0.2], [0.3, 0.4]])  # Not 1D
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="y must have the same size as X."):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)


def test_polyfit1d_w_not_1d():
    x, y, w, x_new = make_test_inputs()

    # valid None input
    model = Polyfit1DModel()
    model.fit(x, y, sample_weight=None, bandwidth=0.1, reg_grid=x_new)
    result = model.predict(x_new)
    assert result.shape == (len(x_new),)

    w = np.array([[1.0, 1.0], [1.0, 1.0]])  # Not 1D
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="sample_weight must have the same length as y"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)


def test_polyfit1d_x_y_size_mismatch():
    x, y, w, x_new = make_test_inputs()
    y = np.array([0.1, 0.2, 0.3])  # Different size than x
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="y must have the same size as X."):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)


def test_polyfit1d_w_negative():
    x, y, w, x_new = make_test_inputs()
    w = np.array([-1.0, 1.0, 1.0, 2.0, 3.0])  # Negative weight
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="All sample weights must be non-negative"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)


def test_polyfit1d_y_w_size_mismatch():
    x, y, w, x_new = make_test_inputs()
    w = np.array([1.0, 1.0, 1.0])  # Different size than y
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="sample_weight must have the same length as y"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)


def test_polyfit1d_x_new_not_1d():
    x, y, w, _ = make_test_inputs()
    x_new = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="reg_grid must be a 1D array"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)


def test_polyfit1d_x_new_empty():
    x, y, w, _ = make_test_inputs()
    x_new = np.array([], dtype=np.float64)
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="Found array with 0 sample"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)


def test_polyfit1d_bandwidth_non_positive():
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="bandwidth must be positive"):
        model.fit(x, y, sample_weight=w, bandwidth=0.0, reg_grid=x_new)


def test_polyfit1d_kernel_type_invalid():
    with pytest.raises(ValueError, match="kernel must be one of"):
        Polyfit1DModel(kernel_type="invalid")


def test_polyfit1d_degree_non_positive():
    x, y, w, x_new = make_test_inputs()
    with pytest.raises(ValueError, match="Degree of polynomial, degree, should be positive"):
        Polyfit1DModel(degree=0)


def test_polyfit1d_deriv_negative():
    x, y, w, x_new = make_test_inputs()
    with pytest.raises(ValueError, match="Order of derivative, deriv, should be positive"):
        Polyfit1DModel(deriv=-1)


def test_polyfit1d_degree_less_than_deriv():
    x, y, w, x_new = make_test_inputs()
    with pytest.raises(ValueError, match="Degree of polynomial, degree, should be greater than or equal to order of derivative, deriv"):
        Polyfit1DModel(degree=1, deriv=2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit1d_nan_inputs(dtype):
    # Test NaN in X
    x = np.array([0.1, np.nan, 0.8], dtype=dtype)
    y = np.array([0.01, 0.04, 0.64], dtype=dtype)
    w = np.ones_like(x)
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1)

    # Test NaN in y
    x = np.array([0.1, 0.5, 0.8], dtype=dtype)
    y_nan = np.array([0.01, np.nan, 0.64], dtype=dtype)
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.fit(x, y_nan, sample_weight=w, bandwidth=0.1)

    # Test NaN in sample_weight
    w_nan = np.array([1.0, np.nan, 1.0], dtype=dtype)
    y = np.array([0.01, 0.04, 0.64], dtype=dtype)
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.fit(x, y, sample_weight=w_nan, bandwidth=0.1)

    # Test NaN in prediction input
    model.fit(x, y, sample_weight=w, bandwidth=0.1)
    x_new_nan = np.array([0.1, np.nan, 0.8], dtype=dtype)
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.predict(x_new_nan)


@pytest.mark.parametrize("bad_type", [float("nan"), np.nan])
def test_polyfit1d_bandwidth_nan(bad_type):
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="bandwidth.*NaN|bandwidth must be positive"):
        model.fit(x, y, sample_weight=w, bandwidth=bad_type, reg_grid=x_new)


@pytest.mark.parametrize("bad_type", ["2", [1], (2,), {"a": 1}])
def test_polyfit1d_bandwidth_non_numeric_type(bad_type):
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel()
    with pytest.raises((TypeError, ValueError), match="bandwidth.*should be.*number|bandwidth must be positive"):
        model.fit(x, y, sample_weight=w, bandwidth=bad_type, reg_grid=x_new)


@pytest.mark.parametrize("bad_type", [1.5, "2", float("nan"), np.nan, None, [1], (2,), {"a": 1}])
def test_polyfit1d_degree_non_int_type(bad_type):
    with pytest.raises((TypeError, ValueError), match="degree.*should be.*int|Invalid value.*degree"):
        Polyfit1DModel(degree=bad_type)


@pytest.mark.parametrize("bad_type", [1.5, "2", float("nan"), np.nan, None, [1], (2,), {"a": 1}])
def test_polyfit1d_deriv_non_int_type(bad_type):
    with pytest.raises((TypeError, ValueError), match="deriv.*should be.*int|Invalid value.*deriv"):
        Polyfit1DModel(deriv=bad_type)


def test_polyfit1d_model_predict_wrong_2d_input():
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel()
    model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)

    # Test with 2D input
    with pytest.raises(ValueError, match="X must have exactly 1 feature"):
        model.predict(x_new.reshape(-1, 5))


def test_polyfit1d_model_predict_2d_input():
    x, y, w, x_new = make_test_inputs()
    # Test with 2D input
    model = Polyfit1DModel()
    model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)
    y_pred = model.predict(x_new.reshape(-1, 1))
    assert y_pred.shape == (len(x_new),)
    assert np.all(np.isfinite(y_pred)), "Prediction contains NaN or Inf values"

    y_pred = model.predict(x_new.reshape(-1, 1), use_model_interp=False)
    assert y_pred.shape == (len(x_new),)
    assert np.all(np.isfinite(y_pred)), "Prediction contains NaN or Inf values"


def test_polyfit1d_model_predict_without_fit():
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="This .* instance is not fitted yet|Model must be fitted"):
        model.predict(x_new)


def test_polyfit1d_model_get_fitted_grids():
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel()
    model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=x_new)

    # Test if fitted grids are returned correctly
    obs_grid, obs_fitted_values = model.get_fitted_grids()
    assert obs_grid.shape == (len(x_new),)
    assert obs_fitted_values.shape == (len(x_new),)
    assert np.all(np.isfinite(obs_fitted_values)), "Fitted values contain NaN or Inf"


def test_polyfit1d_model_custom_reg_grid():
    x, y, w, x_new = make_test_inputs()
    # Test with custom regression grid
    reg_grid = np.array([0.1, 0.15, 0.2])
    model = Polyfit1DModel()
    model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=reg_grid)
    y_pred = model.predict(x_new)
    assert len(y_pred) == len(x_new)


def test_polyfit1d_model_reg_grid_out_of_range():
    x, y, w, _ = make_test_inputs()
    # Test with regression grid outside input range
    reg_grid = np.array([0.0, 0.3])  # Outside [0.1, 0.2]
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="reg_grid must be within the range of input X"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=reg_grid)


def test_polyfit1d_model_reg_grid_wrong_shape():
    x, y, w, _ = make_test_inputs()
    # Test with 2D regression grid
    reg_grid = np.array([[0.1, 0.15], [0.2, 0.25]])
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="reg_grid must be a 1D array"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=reg_grid)


def test_polyfit1d_model_reg_grid_too_few_points():
    x, y, w, _ = make_test_inputs()
    # Test with regression grid with too few points
    reg_grid = np.array([0.15])  # Only 1 point
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="reg_grid must have at least 2 points"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=reg_grid)


def test_polyfit1d_model_reg_grid_with_nan():
    x, y, w, _ = make_test_inputs()
    # Test with regression grid containing NaN
    reg_grid = np.array([0.1, np.nan, 0.2])
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="Input contains NaN"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, reg_grid=reg_grid)


@pytest.mark.parametrize("bad_num", [None, 1.5, [1.5], {'a': 1.5}])
def test_polyfit1d_model_bad_num_points_reg_grid(bad_num):
    x, y, w, _ = make_test_inputs()
    model = Polyfit1DModel()
    with pytest.raises(TypeError, match="Number of points for interpolation grid, num_points_reg_grid, should be an integer"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, num_points_reg_grid=bad_num)


def test_polyfit1d_wrong_interp_kind():
    x, y, w, x_new = make_test_inputs()
    with pytest.raises(ValueError, match="interp_kind must be one of"):
        model = Polyfit1DModel(interp_kind="invalid")


def test_polyfit1d_wrong_bandwidth_selection_method():
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="bandwidth_selection_method must be one of"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, bandwidth_selection_method="invalid", reg_grid=x_new)


def test_polyfit1d_num_bw_candidates():
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel()
    with pytest.raises(TypeError, match="Number of bandwidth candidates, num_bw_candidates, should be an integer"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, num_bw_candidates=1.5, reg_grid=x_new)
    with pytest.raises(ValueError, match="Number of bandwidth candidates, num_bw_candidates, should be at least 2"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, num_bw_candidates=1, reg_grid=x_new)


def test_polyfit1d_cv_folds():
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel()
    with pytest.raises(ValueError, match="Number of cross-validation folds, cv_folds, should be at least 2"):
        model.fit(x, y, sample_weight=w, bandwidth=0.1, bandwidth_selection_method="cv", cv_folds=1)

    with pytest.raises(TypeError, match="Number of cross-validation folds, cv_folds, should be an integer"):
        model.fit(x, y, sample_weight=w, bandwidth_selection_method="cv", cv_folds=2.5, reg_grid=x_new)


def test_polyfit1d_custom_bw_candidates():
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel(random_seed=100, kernel_type=KernelType.EPANECHNIKOV)
    model.fit(x, y, sample_weight=w, custom_bw_candidates=np.array([[0.1], [0.2]]))
    assert model.bandwidth_ == np.float64(0.2)
    with pytest.raises(ValueError, match="All CV scores are non-finite."):
        model.fit(x, y, sample_weight=w, bandwidth_selection_method="cv", custom_bw_candidates=np.array([[0.005], [0.006]]), reg_grid=x_new)
    with pytest.raises(ValueError, match="All GCV scores are non-finite."):
        model.fit(x, y, sample_weight=w, custom_bw_candidates=np.array([0.005, 0.006]), reg_grid=x_new)
    with pytest.raises(ValueError, match="custom_bw_candidates must have exactly 1 feature"):
        model.fit(x, y, sample_weight=w, custom_bw_candidates=np.array([[0.1, 0.2]]), reg_grid=x_new)


def test_polyfit1d_gcv_bandwidth_selection():
    x = np.concatenate([np.linspace(0, 1, 21), np.linspace(0, 1, 21)])
    y = x**2 - x * 3 + 0.5
    w = np.ones_like(x)
    x_new = np.linspace(0, 1, 21)
    model = Polyfit1DModel(random_seed=100, kernel_type=KernelType.EPANECHNIKOV)
    model.fit(x, y, sample_weight=w, bandwidth_selection_method="gcv", reg_grid=x_new)
    assert math.fabs(model.bandwidth_ - 0.15) < 1e-5
    assert "bandwidth_selection_results_" in model.__dict__, "bandwidth_selection_results_ should be set after fitting with gcv method"


def test_polyfit1d_cv_bandwidth_selection():
    x = np.concatenate([np.linspace(0, 1, 21), np.linspace(0, 1, 21)])
    y = x**2 - x * 3 + 0.5
    w = np.ones_like(x)
    x_new = np.linspace(0, 1, 21)
    model = Polyfit1DModel(random_seed=100, kernel_type=KernelType.EPANECHNIKOV)
    model.fit(x, y, sample_weight=w, bandwidth_selection_method="cv", cv_folds=5, reg_grid=x_new)
    assert math.fabs(model.bandwidth_ - 0.15) < 1e-5
    assert "bandwidth_selection_results_" in model.__dict__, "bandwidth_selection_results_ should be set after fitting with cv method"


def test_polyfit1d_unable_generate_bandwidth_candidates():
    x = np.array([0.1, 0.2, 0.2])
    y = np.array([0.1, 0.2, 0.3])
    w = np.array([1.0, 1.0, 1.0])
    model = Polyfit1DModel(random_seed=100)
    with pytest.raises(ValueError, match="Not enough unique support points"):
        model.fit(x, y, sample_weight=w, reg_grid=x)


def test_polyfit1d_polyfit_fail(monkeypatch):
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel(random_seed=100)

    import pflm.smooth.polyfit_model_1d as pm

    def fake_polyfit1d(x, y, w, x_new, bandwidth, kernel_type, degree, deriv):
        raise ValueError("Error during polynomial fitting")

    monkeypatch.setattr(pm, "polyfit1d_f32", fake_polyfit1d, raising=True)
    monkeypatch.setattr(pm, "polyfit1d_f64", fake_polyfit1d, raising=True)
    with pytest.raises(ValueError, match="Error in polyfit1d"):
        model.fit(x, y, sample_weight=w, bandwidth=0.5, reg_grid=x_new)


def test_polyfit1d_polyfit_predict_fail(monkeypatch):
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel(random_seed=100)
    model.fit(x, y, sample_weight=w, reg_grid=x_new)

    def fake_polyfit1d(x, y, w, x_new, bandwidth, kernel_type, degree, deriv):
        raise ValueError("Error during polynomial fitting")

    monkeypatch.setattr(model, "_polyfit1d_func", fake_polyfit1d, raising=True)
    with pytest.raises(ValueError, match="Error during polynomial fitting"):
        model.predict(x_new, use_model_interp=False)


def test_polyfit1d_predict_interp_fail(monkeypatch):
    x, y, w, x_new = make_test_inputs()
    model = Polyfit1DModel(random_seed=100)
    model.fit(x, y, sample_weight=w, reg_grid=x_new)

    import pflm.smooth.polyfit_model_1d as pm

    def fake_interp1d(x, y, x_new, method):
        raise ValueError("Error during interpolation")

    monkeypatch.setattr(pm, "interp1d", fake_interp1d, raising=True)
    with pytest.raises(ValueError, match="Error during interpolation"):
        model.predict(x_new, use_model_interp=True)
