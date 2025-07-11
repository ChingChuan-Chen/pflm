import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.smooth.kernel import KernelType
from pflm.smooth.polyfit import polyfit1d


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit1d_happy_case(dtype):
    bw = 0.25
    x = np.linspace(0.0, 1.0, 11, dtype=dtype)
    y = 2.0 * x**2 + 3 * x
    w = np.ones_like(x)
    x_new = np.linspace(0.0, 1.0, 11, dtype=dtype)

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.GAUSSIAN, 1, 0),
        np.array(
            [
                -0.059162939216,
                0.325036778510,
                0.731014910930,
                1.161759855632,
                1.619430366873,
                2.105204388503,
                2.619430366873,
                3.161759855632,
                3.731014910930,
                4.325036778510,
                4.940837060784,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.LOGISTIC, 1, 0),
        np.array(
            [
                -0.161150328985,
                0.277047311686,
                0.724601740286,
                1.184564011983,
                1.659446412861,
                2.150885243892,
                2.659446412861,
                3.184564011983,
                3.724601740286,
                4.277047311686,
                4.838849671015,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.SIGMOID, 1, 0),
        np.array(
            [
                -0.130518482307,
                0.294439062982,
                0.727032287138,
                1.173968172887,
                1.640231007870,
                2.128680341675,
                2.640231007870,
                3.173968172887,
                3.727032287138,
                4.294439062982,
                4.869481517693,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.GAUSSIAN_VAR, 1, 0),
        np.array(
            [
                -0.026638964783,
                0.336983831767,
                0.725923679570,
                1.143210394558,
                1.591704222727,
                2.074075166076,
                2.591704222727,
                3.143210394558,
                3.725923679570,
                4.336983831767,
                4.973361035217,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.RECTANGULAR, 1, 0),
        np.array(
            [
                -0.006666666667,
                0.340000000000,
                0.720000000000,
                1.120000000000,
                1.560000000000,
                2.040000000000,
                2.560000000000,
                3.120000000000,
                3.720000000000,
                4.340000000000,
                4.993333333333,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.TRIANGULAR, 1, 0),
        np.array(
            [
                -0.003157894737,
                0.334482758621,
                0.701538461538,
                1.101538461538,
                1.541538461538,
                2.021538461538,
                2.541538461538,
                3.101538461538,
                3.701538461538,
                4.334482758621,
                4.996842105263,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.EPANECHNIKOV, 1, 0),
        np.array(
            [
                -0.004684014870,
                0.337087794433,
                0.706823529412,
                1.106823529412,
                1.546823529412,
                2.026823529412,
                2.546823529412,
                3.106823529412,
                3.706823529412,
                4.337087794433,
                4.995315985130,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.BIWEIGHT, 1, 0),
        np.array(
            [
                -0.002780677479,
                0.334288436982,
                0.698334331935,
                1.098334331935,
                1.538334331935,
                2.018334331935,
                2.538334331935,
                3.098334331935,
                3.698334331935,
                4.334288436982,
                4.997219322521,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.TRIWEIGHT, 1, 0),
        np.array(
            [
                -0.001370698495,
                0.332100791473,
                0.693680101109,
                1.093680101109,
                1.533680101109,
                2.013680101109,
                2.533680101109,
                3.093680101109,
                3.693680101109,
                4.332100791473,
                4.998629301505,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.TRICUBE, 1, 0),
        np.array(
            [
                -0.002761917832,
                0.334665254777,
                0.697892313673,
                1.097892313673,
                1.537892313673,
                2.017892313673,
                2.537892313673,
                3.097892313673,
                3.697892313673,
                4.334665254777,
                4.997238082168,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(x, y, w, x_new, bw, KernelType.COSINE, 1, 0),
        np.array(
            [
                -0.004357137153,
                0.336602913011,
                0.705278640450,
                1.105278640450,
                1.545278640450,
                2.025278640450,
                2.545278640450,
                3.105278640450,
                3.705278640450,
                4.336602913011,
                4.995642862847,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


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

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.GAUSSIAN, 1, 0),
        np.array(
            [
                0.114586353967,
                0.159087772528,
                0.203476488507,
                0.247741547666,
                0.291870963774,
                0.335851693572,
                0.379669608445,
                0.423309462831,
                0.466754859524,
                0.509988212162,
                0.552990705324,
                0.595742252773,
                0.638221454523,
                0.680405553527,
                0.722270392867,
                0.763790374458,
                0.804938420320,
                0.845685937585,
                0.886002788393,
                0.925857265912,
                0.965216077679,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.GAUSSIAN, 3, 1),
        np.array(
            [
                0.066008431547,
                0.074644211399,
                0.083340851498,
                0.091848185525,
                0.099909859941,
                0.107263002444,
                0.113637855948,
                0.118757379937,
                0.122336821122,
                0.124083255320,
                0.123695102554,
                0.120861617369,
                0.115262356409,
                0.106566625290,
                0.094432906872,
                0.078508272999,
                0.058427781819,
                0.033813862791,
                0.004275691486,
                -0.030591443704,
                -0.071206780839,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.LOGISTIC, 1, 0),
        np.array(
            [
                0.120222650606,
                0.163744104849,
                0.207198694246,
                0.250579480300,
                0.293879147177,
                0.337090038203,
                0.380204207606,
                0.423213488598,
                0.466109578285,
                0.508884139084,
                0.551528915463,
                0.594035863853,
                0.636397292677,
                0.678606008587,
                0.720655464275,
                0.762539902773,
                0.804254492873,
                0.845795450406,
                0.887160140438,
                0.928347156115,
                0.969356370812,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.LOGISTIC, 3, 1),
        np.array(
            [
                0.047653739505,
                0.062763013359,
                0.076516307075,
                0.088820556267,
                0.099572292135,
                0.108657906240,
                0.115954088956,
                0.121328443314,
                0.124640271990,
                0.125741531166,
                0.124477940845,
                0.120690237049,
                0.114215547083,
                0.104888864930,
                0.092544599833,
                0.077018167527,
                0.058147590613,
                0.035775072501,
                0.009748508441,
                -0.020077102260,
                -0.053838376686,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.SIGMOID, 1, 0),
        np.array(
            [
                0.117535400789,
                0.161376166822,
                0.205175309934,
                0.248924320544,
                0.292613214023,
                0.336229972379,
                0.379759957990,
                0.423185407269,
                0.466485124188,
                0.509634482161,
                0.552605810613,
                0.595369198442,
                0.637893700107,
                0.680148887311,
                0.722106650574,
                0.763743117971,
                0.805040522972,
                0.845988826274,
                0.886586890384,
                0.926843033215,
                0.966774853216,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.SIGMOID, 3, 1),
        np.array(
            [
                0.057782941217,
                0.069730188707,
                0.080767699437,
                0.090846692743,
                0.099887899826,
                0.107779557398,
                0.114376871942,
                0.119502913112,
                0.122950715559,
                0.124486286307,
                0.123852263290,
                0.120772129546,
                0.114955082781,
                0.106101791453,
                0.093911252319,
                0.078088773510,
                0.058354787706,
                0.034453856099,
                0.006162977115,
                -0.026701744137,
                -0.064280850759,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.GAUSSIAN_VAR, 1, 0),
        np.array(
            [
                0.110947314133,
                0.156101200364,
                0.201097958085,
                0.245932441500,
                0.290595814726,
                0.335075970960,
                0.379357751493,
                0.423423014262,
                0.467250579590,
                0.510816066137,
                0.554091619151,
                0.597045523594,
                0.639641685033,
                0.681838949698,
                0.723590220021,
                0.764841300849,
                0.805529380496,
                0.845581003914,
                0.884909322265,
                0.923410286498,
                0.960957260858,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.GAUSSIAN_VAR, 3, 1),
        np.array(
            [
                0.081800971835,
                0.084431585507,
                0.088703128302,
                0.094041896410,
                0.099946779745,
                0.105961664339,
                0.111657921503,
                0.116622510671,
                0.120449320547,
                0.122732391605,
                0.123060180651,
                0.121010293436,
                0.116144235712,
                0.108001765250,
                0.096094381909,
                0.079897359176,
                0.058839458991,
                0.032288993188,
                -0.000465987307,
                -0.040247286035,
                -0.088033283274,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.RECTANGULAR, 1, 0),
        np.array(
            [
                0.095238095238,
                0.144095238095,
                0.205000000000,
                0.246785714286,
                0.286428571429,
                0.333035714286,
                0.381388888889,
                0.430138888889,
                0.465454545455,
                0.507727272727,
                0.550000000000,
                0.592272727273,
                0.634545454545,
                0.677916666667,
                0.719166666667,
                0.760119047619,
                0.799761904762,
                0.844821428571,
                0.888214285714,
                0.908142857143,
                0.942857142857,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.RECTANGULAR, 3, 1),
        np.array(
            [
                0.228835978836,
                0.162934193122,
                0.095551587302,
                0.105328373016,
                0.092134199134,
                0.085761634199,
                0.089987373737,
                0.089731691919,
                0.126868686869,
                0.127353826729,
                0.125242812743,
                0.120535644911,
                0.113232323232,
                0.119874188312,
                0.104496753247,
                0.110082972583,
                0.072544011544,
                0.023156746032,
                -0.023182539683,
                -0.136672619048,
                -0.258333333333,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.TRIANGULAR, 1, 0),
        np.array(
            [
                0.109949225918,
                0.153743661968,
                0.200425885998,
                0.246268779525,
                0.291103017639,
                0.334323030918,
                0.376867836726,
                0.422017725182,
                0.469878907339,
                0.515026357946,
                0.560186757216,
                0.605102364700,
                0.647267381681,
                0.688441662153,
                0.731396913225,
                0.774743306363,
                0.815308401128,
                0.851473658933,
                0.881547049091,
                0.897209516895,
                0.907550429225,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.TRIANGULAR, 3, 1),
        np.array(
            [
                0.215612371249,
                0.159278528932,
                0.110802207787,
                0.101092980854,
                0.093596127066,
                0.087469241029,
                0.087501396280,
                0.090925422490,
                0.097644252280,
                0.113528316737,
                0.119606182650,
                0.122626032417,
                0.127979340998,
                0.126831520357,
                0.124062157970,
                0.111581482574,
                0.078532601026,
                0.032409268387,
                -0.038837704611,
                -0.176943052829,
                -0.364122194033,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.EPANECHNIKOV, 1, 0),
        np.array(
            [
                0.111069479239,
                0.152700799416,
                0.199552902871,
                0.244701570772,
                0.288410270577,
                0.332966817520,
                0.378635913259,
                0.425844450664,
                0.473212365964,
                0.516403786619,
                0.559346656993,
                0.602050771672,
                0.644521566603,
                0.686133188092,
                0.727155392828,
                0.767187169989,
                0.806377826185,
                0.843945256986,
                0.876817386601,
                0.901555255376,
                0.925257248143,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.EPANECHNIKOV, 3, 1),
        np.array(
            [
                0.229396750139,
                0.164413835612,
                0.106915594550,
                0.101274137788,
                0.095945565125,
                0.090344514186,
                0.088389016349,
                0.089172508448,
                0.095784963718,
                0.113962292181,
                0.120314624390,
                0.123715107751,
                0.130727954314,
                0.128913098253,
                0.123115157210,
                0.108450169453,
                0.073252337907,
                0.028718130837,
                -0.032377921384,
                -0.163308364042,
                -0.339602117920,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.BIWEIGHT, 1, 0),
        np.array(
            [
                0.115002887296,
                0.157070554663,
                0.200397387109,
                0.244719374045,
                0.288803798957,
                0.332885288999,
                0.377481412777,
                0.423125519980,
                0.469837762156,
                0.516213134251,
                0.561434287132,
                0.605750136065,
                0.649301372036,
                0.692022926066,
                0.733724377854,
                0.773923106409,
                0.811829372417,
                0.845811328971,
                0.874139006550,
                0.896592667170,
                0.911781546260,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.BIWEIGHT, 3, 1),
        np.array(
            [
                0.214404272682,
                0.163061635479,
                0.113004592424,
                0.095249223874,
                0.092400978870,
                0.090969868595,
                0.089872218543,
                0.090548586930,
                0.093002848846,
                0.104727096712,
                0.116184842517,
                0.125804933490,
                0.134168561613,
                0.134211693244,
                0.129743279955,
                0.110215539750,
                0.081471265493,
                0.037844925751,
                -0.038651929526,
                -0.186831047990,
                -0.418825443215,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.TRIWEIGHT, 1, 0),
        np.array(
            [
                0.113001114942,
                0.158347054201,
                0.201875140491,
                0.245428554724,
                0.289133568341,
                0.332871422471,
                0.376887220403,
                0.421544968989,
                0.467217591174,
                0.513775151351,
                0.560403719915,
                0.606560732993,
                0.651996454949,
                0.696436199142,
                0.739396872140,
                0.780036547324,
                0.817089521692,
                0.849072268501,
                0.874737989373,
                0.892821667321,
                0.900170361889,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.TRIWEIGHT, 3, 1),
        np.array(
            [
                0.180504573483,
                0.158560892349,
                0.114852330664,
                0.088620876181,
                0.082646356514,
                0.086125052727,
                0.089841255546,
                0.093047607530,
                0.096267395721,
                0.102642951162,
                0.113472815155,
                0.124557259004,
                0.132003883765,
                0.134121267850,
                0.129703113621,
                0.116699404672,
                0.095498658622,
                0.050626709052,
                -0.039773080582,
                -0.205910327509,
                -0.470485571637,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.TRICUBE, 1, 0),
        np.array(
            [
                0.118363401843,
                0.158762257829,
                0.199771562010,
                0.243321937291,
                0.287955607484,
                0.332716048590,
                0.377819358749,
                0.423598980559,
                0.470379846068,
                0.517464314195,
                0.563088672918,
                0.606916842973,
                0.649412508465,
                0.690901689713,
                0.731412131864,
                0.770706643877,
                0.807907166399,
                0.841554046993,
                0.871085754202,
                0.897045388613,
                0.916354685006,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.TRICUBE, 3, 1),
        np.array(
            [
                0.219422521870,
                0.166796103059,
                0.112794236615,
                0.091461760435,
                0.092577527050,
                0.094913381629,
                0.092661070237,
                0.090060510926,
                0.090505790673,
                0.098986038715,
                0.114176343004,
                0.128972027254,
                0.138772847197,
                0.139859972003,
                0.130456060562,
                0.107552861617,
                0.080026206501,
                0.039771325160,
                -0.036880177625,
                -0.189356199086,
                -0.458450076277,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.COSINE, 1, 0),
        np.array(
            [
                0.111720970469,
                0.153515068467,
                0.199741687065,
                0.244715775958,
                0.288504167010,
                0.332988833953,
                0.378465124883,
                0.425380293989,
                0.472584377196,
                0.516316591777,
                0.559704711972,
                0.602765664685,
                0.645488660595,
                0.687362556562,
                0.728536926277,
                0.768587792658,
                0.807486919800,
                0.844348452719,
                0.876439450438,
                0.900773955625,
                0.923169695941,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    assert_allclose(
        polyfit1d(xs, ys, ws, x_new, bw, KernelType.COSINE, 3, 1),
        np.array(
            [
                0.227626520700,
                0.164227973706,
                0.108199157019,
                0.100662451738,
                0.095521469870,
                0.090408479833,
                0.088601285009,
                0.089412259329,
                0.095458633155,
                0.112840199507,
                0.119702679140,
                0.123797153521,
                0.131045826247,
                0.129455039268,
                0.123799977245,
                0.108671296838,
                0.074213949107,
                0.029824216402,
                -0.033711867975,
                -0.167377210201,
                -0.349550294468,
            ],
            dtype=dtype,
        ),
        rtol=1e-5,
        atol=1e-6,
    )


def test_polyfit1d_x_not_1d():
    x = np.array([[0.1, 0.2]])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="x must be a 1D array."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_y_not_1d():
    x = np.array([0.1, 0.2])
    y = np.array([[0.1, 0.2]])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="y must be a 1D array."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_w_not_1d():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([[1.0, 1.0]])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="w must be a 1D array."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_x_y_size_mismatch():
    x = np.array([0.1, 0.2])
    y = np.array([0.1])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="y must have the same size as x."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_x_not_sorted():
    x = np.array([0.2, 0.1])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="x must be sorted in ascending order."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_w_negative():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, -1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="All weights in w must be greater than 0."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_x_w_size_mismatch():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="w must have the same size as x."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_xnew_not_1d():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([[0.1, 0.2]])
    with pytest.raises(ValueError, match="x_new must be a 1D array."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_xnew_empty():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([], dtype=np.float64)
    with pytest.raises(ValueError, match="x_new must not be empty."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_xnew_not_strictly_increasing():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.1])
    with pytest.raises(ValueError, match="x_new must be strictly increasing."):
        polyfit1d(x, y, w, x_new, 0.1)


def test_polyfit1d_bandwidth_nonpositive():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="Bandwidth, bandwidth, should be positive."):
        polyfit1d(x, y, w, x_new, 0.0)


def test_polyfit1d_kernel_type_invalid():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.2])

    class Dummy:
        value = 999

    with pytest.raises(ValueError, match="kernel must be one of"):
        polyfit1d(x, y, w, x_new, 0.1, Dummy())


def test_polyfit1d_degree_nonpositive():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="Degree of polynomial, degree, should be positive."):
        polyfit1d(x, y, w, x_new, 0.1, KernelType.GAUSSIAN, 0)


def test_polyfit1d_deriv_negative():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="Order of derivative, deriv, should be positive."):
        polyfit1d(x, y, w, x_new, 0.1, KernelType.GAUSSIAN, 1, -1)


def test_polyfit1d_degree_less_than_deriv():
    x = np.array([0.1, 0.2])
    y = np.array([0.1, 0.2])
    w = np.array([1.0, 1.0])
    x_new = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="Degree of polynomial, degree, should be greater than or equal to order of derivative, deriv."):
        polyfit1d(x, y, w, x_new, 0.1, KernelType.GAUSSIAN, 1, 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_polyfit1d_nan_inputs(dtype):
    x = np.array([0.1, np.nan, 0.8], dtype=dtype)
    y = np.array([0.01, 0.04, 0.64], dtype=dtype)
    w = np.ones_like(x)
    x_new = np.linspace(0.1, 0.8, 5, dtype=dtype)
    with pytest.raises(ValueError, match="Input array x contains NaN values."):
        polyfit1d(x, y, w, x_new, 0.1, KernelType.GAUSSIAN)
    y_nan = np.array([0.01, np.nan, 0.64], dtype=dtype)
    with pytest.raises(ValueError, match="Input array y contains NaN values."):
        polyfit1d(np.array([0.1, 0.5, 0.8], dtype=dtype), y_nan, w, x_new, 0.1, KernelType.GAUSSIAN)
    w_nan = np.array([1.0, np.nan, 1.0], dtype=dtype)
    with pytest.raises(ValueError, match="Input array w contains NaN values."):
        polyfit1d(np.array([0.1, 0.5, 0.8], dtype=dtype), y, w_nan, x_new, 0.1, KernelType.GAUSSIAN)
    x_new_nan = x_new.copy()
    x_new_nan[2] = np.nan
    with pytest.raises(ValueError, match="Input array x_new contains NaN values."):
        polyfit1d(np.array([0.1, 0.5, 0.8], dtype=dtype), y, w, x_new_nan, 0.1, KernelType.GAUSSIAN)
