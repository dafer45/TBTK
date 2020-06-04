#include "TBTK/SlaterKosterState.h"

#include "gtest/gtest.h"

#include <limits>

namespace TBTK{

const double EPSILON_100 = 100*std::numeric_limits<double>::epsilon();

class SlaterKosterStateTest : public ::testing::Test{
protected:
	constexpr static double V_ssS = 1;
	constexpr static double V_spS = 2;
	constexpr static double V_sdS = 3;
	constexpr static double V_ppS = 4;
	constexpr static double V_ppP = 5;
	constexpr static double V_pdS = 6;
	constexpr static double V_pdP = 7;
	constexpr static double V_ddS = 8;
	constexpr static double V_ddP = 9;
	constexpr static double V_ddD = 10;
	constexpr static double E_s = 11;
	constexpr static double E_p = 12;
	constexpr static double E_d = 13;
	class TestParametrization : public SlaterKosterState::Parametrization{
	public:
		virtual TestParametrization* clone() const{
			return new TestParametrization();
		}

		virtual std::complex<double> getParameter(
			double distance,
			SlaterKosterState::Parametrization::Orbital orbital0,
			SlaterKosterState::Parametrization::Orbital orbital1,
			SlaterKosterState::Parametrization::Bond bond
		) const{
			double R = 2 - distance;
			switch(orbital0){
			case SlaterKosterState::Parametrization::Orbital::s:
				switch(orbital1){
				case SlaterKosterState::Parametrization::Orbital::s:
					switch(bond){
					case SlaterKosterState::Parametrization::Bond::Sigma:
						return V_ssS*R;
					default:
						TBTKExit(
							"SlaterKosterStateTest::TestParametrization::operator()",
							"Unknown orbital.",
							"This should never"
							<< " happen, contact"
							<< " the developer."
						);
					}
				case SlaterKosterState::Parametrization::Orbital::p:
					switch(bond){
					case SlaterKosterState::Parametrization::Bond::Sigma:
						return V_spS*R;
					default:
						TBTKExit(
							"SlaterKosterStateTest::TestParametrization::operator()",
							"Unknown orbital.",
							"This should never"
							<< " happen, contact"
							<< " the developer."
						);
					}
				case SlaterKosterState::Parametrization::Orbital::d:
					switch(bond){
					case SlaterKosterState::Parametrization::Bond::Sigma:
						return V_sdS*R;
					default:
						TBTKExit(
							"SlaterKosterStateTest::TestParametrization::operator()",
							"Unknown orbital.",
							"This should never"
							<< " happen, contact"
							<< " the developer."
						);
					}
				default:
					TBTKExit(
						"SlaterKosterStateTest::TestParametrization::operator()",
						"Unknown orbital.",
						"This should never happen, contact the"
						<< " developer."
					);
				}
			case SlaterKosterState::Parametrization::Orbital::p:
				switch(orbital1){
				case SlaterKosterState::Parametrization::Orbital::p:
					switch(bond){
					case SlaterKosterState::Parametrization::Bond::Sigma:
						return V_ppS*R;
					case SlaterKosterState::Parametrization::Bond::Pi:
						return V_ppP*R;
					default:
						TBTKExit(
							"SlaterKosterStateTest::TestParametrization::operator()",
							"Unknown orbital.",
							"This should never"
							<< " happen, contact"
							<< " the developer."
						);
					}
				case SlaterKosterState::Parametrization::Orbital::d:
					switch(bond){
					case SlaterKosterState::Parametrization::Bond::Sigma:
						return V_pdS*R;
					case SlaterKosterState::Parametrization::Bond::Pi:
						return V_pdP*R;
					default:
						TBTKExit(
							"SlaterKosterStateTest::TestParametrization::operator()",
							"Unknown orbital.",
							"This should never"
							<< " happen, contact"
							<< " the developer."
						);
					}
				default:
					TBTKExit(
						"SlaterKosterStateTest::TestParametrization::operator()",
						"Unknown orbital.",
						"This should never happen, contact the"
						<< " developer."
					);
				}
			case SlaterKosterState::Parametrization::Orbital::d:
				switch(orbital1){
				case SlaterKosterState::Parametrization::Orbital::d:
					switch(bond){
					case SlaterKosterState::Parametrization::Bond::Sigma:
						return V_ddS*R;
					case SlaterKosterState::Parametrization::Bond::Pi:
						return V_ddP*R;
					case SlaterKosterState::Parametrization::Bond::Delta:
						return V_ddD*R;
					default:
						TBTKExit(
							"SlaterKosterStateTest::TestParametrization::operator()",
							"Unknown orbital.",
							"This should never"
							<< " happen, contact"
							<< " the developer."
						);
					}
				default:
					TBTKExit(
						"SlaterKosterStateTest::TestParametrization::operator()",
						"Unknown orbital.",
						"This should never happen, contact the"
						<< " developer."
					);
				}
			default:
				TBTKExit(
					"SlaterKosterStateTest::TestParametrization::operator()",
					"Unknown orbital.",
					"This should never happen, contact the"
					<< " developer."
				);
			}
		}

		virtual std::complex<double> getOnSiteTerm(Orbital orbital) const{
			switch(orbital){
			case Orbital::s:
				return E_s;
			case Orbital::p:
				return E_p;
			case Orbital::d:
				return E_d;
			default:
				TBTKExit(
					"SlaterKosterStateTest::TestParametirzation::getOnSiteTerm()",
					"Unknown orbital.",
					"This should never happen, contact the"
					<< " developer."
				);
			}
		}
	};

	std::vector<Vector3d> positions;
	SlaterKosterState s[4][1];
	SlaterKosterState p[4][3];
	SlaterKosterState d[4][5];
	void SetUp() override{
		positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
		for(unsigned int n = 0; n < positions.size(); n++){
			s[n][0] = SlaterKosterState(
				positions[n],
				"s",
				TestParametrization()
			);
			p[n][0] = SlaterKosterState(
				positions[n],
				"x",
				TestParametrization()
			);
			p[n][1] = SlaterKosterState(
				positions[n],
				"y",
				TestParametrization()
			);
			p[n][2] = SlaterKosterState(
				positions[n],
				"z",
				TestParametrization()
			);
			d[n][0] = SlaterKosterState(
				positions[n],
				"xy",
				TestParametrization()
			);
			d[n][1] = SlaterKosterState(
				positions[n],
				"yz",
				TestParametrization()
			);
			d[n][2] = SlaterKosterState(
				positions[n],
				"zx",
				TestParametrization()
			);
			d[n][3] = SlaterKosterState(
				positions[n],
				"x^2-y^2",
				TestParametrization()
			);
			d[n][4] = SlaterKosterState(
				positions[n],
				"3z^2-r^2",
				TestParametrization()
			);
		}
	}

	double radialFunction(
		const Vector3d &position0,
		const Vector3d &position1
	){
		return 2 - (position1 - position0).norm();
	}

	double getL(const Vector3d &position0, const Vector3d &position1){
		Vector3d difference = position0 - position1;
		return difference.x/difference.norm();
	}

	double getM(const Vector3d &position0, const Vector3d &position1){
		Vector3d difference = position0 - position1;
		return difference.y/difference.norm();
	}

	double getN(const Vector3d &position0, const Vector3d &position1){
		Vector3d difference = position0 - position1;
		return difference.z/difference.norm();
	}
};

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.0 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement0){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= s[i][0].getMatrixElement(s[j][0]);
			if(i == j){
				EXPECT_NEAR(
					real(result),
					E_s,
					EPSILON_100
				);
			}
			else{
				EXPECT_NEAR(
					real(result),
					V_ssS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.1 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement1){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][0].getMatrixElement(s[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					l*V_spS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.2 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement2){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][1].getMatrixElement(s[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					m*V_spS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.3 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement3){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][2].getMatrixElement(s[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					n*V_spS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.4 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement4){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][0].getMatrixElement(s[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*l*m*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.5 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement5){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][1].getMatrixElement(s[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*m*n*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.6 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement6){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][2].getMatrixElement(s[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*n*l*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.7 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement7){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][3].getMatrixElement(s[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)/2.)*(l*l - m*m)*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.8 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement8){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][4].getMatrixElement(s[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(n*n - (l*l + m*m)/2.)*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.9 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement9){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= s[i][0].getMatrixElement(p[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-l*V_spS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.10 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement10){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][0].getMatrixElement(p[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), E_p, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(l*l*V_ppS + (1 - l*l)*V_ppP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.11 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement11){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][1].getMatrixElement(p[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(l*m*V_ppS - l*m*V_ppP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.12 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement12){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][2].getMatrixElement(p[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(l*n*V_ppS - l*n*V_ppP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.13 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement13){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][0].getMatrixElement(p[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0 , EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)*l*l*m*V_pdS + m*(1 - 2*l*l)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.14 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement14){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][1].getMatrixElement(p[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)*l*m*n*V_pdS - 2*l*m*n*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.15 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement15){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][2].getMatrixElement(p[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)*l*l*n*V_pdS + n*(1 - 2*l*l)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.16 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement16){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][3].getMatrixElement(p[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((sqrt(3.)/2.)*l*(l*l - m*m)*V_pdS + l*(1 - l*l + m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.17 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement17){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][4].getMatrixElement(p[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(l*(n*n - (l*l + m*m)/2.)*V_pdS - sqrt(3.)*l*n*n*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.18 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement18){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= s[i][0].getMatrixElement(p[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-m*V_spS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.19 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement19){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][0].getMatrixElement(p[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(l*m*V_ppS - l*m*V_ppP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.20 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement20){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][1].getMatrixElement(p[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), E_p, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(m*m*V_ppS + (1 - m*m)*V_ppP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.21 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement21){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][2].getMatrixElement(p[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(m*n*V_ppS - m*n*V_ppP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.22 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement22){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][0].getMatrixElement(p[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)*m*m*l*V_pdS + l*(1 - 2*m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.23 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement23){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][1].getMatrixElement(p[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)*m*m*n*V_pdS + n*(1 - 2*m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.24 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement24){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][2].getMatrixElement(p[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)*n*l*m*V_pdS - 2*n*l*m*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.25 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement25){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][3].getMatrixElement(p[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((sqrt(3.)/2.)*m*(l*l - m*m)*V_pdS - m*(1 + l*l - m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.26 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement26){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][4].getMatrixElement(p[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(m*(n*n - (l*l + m*m)/2.)*V_pdS - sqrt(3.)*m*n*n*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.27 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement27){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= s[i][0].getMatrixElement(p[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-n*V_spS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.28 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement28){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][0].getMatrixElement(p[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(l*n*V_ppS - l*n*V_ppP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.29 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement29){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][1].getMatrixElement(p[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(n*m*V_ppS - n*m*V_ppP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.30 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement30){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][2].getMatrixElement(p[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), E_p, EPSILON_100);
			}
			else{
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(n*n*V_ppS + (1 - n*n)*V_ppP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.31 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement31){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][0].getMatrixElement(p[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)*n*l*m*V_pdS - 2*n*l*m*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.32 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement32){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][1].getMatrixElement(p[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)*n*n*m*V_pdS + m*(1 - 2*n*n)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.33 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement33){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][2].getMatrixElement(p[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)*n*n*l*V_pdS + l*(1 - 2*n*n)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.34 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement34){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][3].getMatrixElement(p[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((sqrt(3.)/2.)*n*(l*l - m*m)*V_pdS - n*(l*l - m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.35 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement35){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][4].getMatrixElement(p[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(n*(n*n - (l*l + m*m)/2.)*V_pdS + sqrt(3.)*n*(l*l + m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.36 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement36){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= s[i][0].getMatrixElement(d[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*l*m*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.37 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement37){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][0].getMatrixElement(d[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(sqrt(3.)*l*l*m*V_pdS + m*(1 - 2*l*l)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.38 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement38){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][1].getMatrixElement(d[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(sqrt(3.)*m*m*l*V_pdS + l*(1 - 2*m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.39 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement39){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][2].getMatrixElement(d[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(sqrt(3.)*n*l*m*V_pdS - 2*n*l*m*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.40 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement40){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][0].getMatrixElement(d[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), E_d, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(3*l*l*m*m*V_ddS + (l*l + m*m - 4*l*l*m*m)*V_ddP + (n*n + l*l*m*m)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.41 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement41){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][1].getMatrixElement(d[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(3*l*m*m*n*V_ddS + l*n*(1 - 4*m*m)*V_ddP + l*n*(m*m - 1)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.42 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement42){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][2].getMatrixElement(d[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(3*l*l*m*n*V_ddS + m*n*(1 - 4*l*l)*V_ddP + m*n*(l*l - 1)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.43 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement43){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][3].getMatrixElement(d[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((3./2.)*l*m*(l*l - m*m)*V_ddS + 2*l*m*(m*m - l*l)*V_ddP + (l*m*(l*l - m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.44 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement44){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][4].getMatrixElement(d[j][0]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*(l*m*(n*n - (l*l + m*m)/2.)*V_ddS - 2*l*m*n*n*V_ddP + (l*m*(1 + n*n)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.45 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement45){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= s[i][0].getMatrixElement(d[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*m*n*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.46 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement46){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][0].getMatrixElement(d[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(sqrt(3.)*l*m*n*V_pdS - 2*l*m*n*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.47 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement47){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][1].getMatrixElement(d[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(sqrt(3.)*m*m*n*V_pdS + n*(1 - 2*m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.48 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement48){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][2].getMatrixElement(d[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(sqrt(3.)*n*n*m*V_pdS + m*(1 - 2*n*n)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.49 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement49){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][0].getMatrixElement(d[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(3*l*m*m*n*V_ddS + l*n*(1 - 4*m*m)*V_ddP + l*n*(m*m - 1)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.50 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement50){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][1].getMatrixElement(d[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), E_d, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(3*m*m*n*n*V_ddS + (m*m + n*n - 4*m*m*n*n)*V_ddP + (l*l + m*m*n*n)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.51 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement51){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][2].getMatrixElement(d[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(3*m*n*n*l*V_ddS + m*l*(1 - 4*n*n)*V_ddP + m*l*(n*n - 1)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.52 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement52){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][3].getMatrixElement(d[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((3./2.)*m*n*(l*l - m*m)*V_ddS + m*n*(1 + 2*(l*l - m*m))*V_ddP + m*n*(1 + (l*l - m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.53 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement53){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][4].getMatrixElement(d[j][1]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*(m*n*(n*n - (l*l + m*m)/2.)*V_ddS + m*n*(l*l + m*m - n*n)*V_ddP - (m*n*(l*l + m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.54 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement54){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= s[i][0].getMatrixElement(d[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*n*l*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.55 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement55){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][0].getMatrixElement(d[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(sqrt(3.)*l*l*n*V_pdS + n*(1 - 2*l*l)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.56 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement56){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][1].getMatrixElement(d[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(sqrt(3.)*m*n*l*V_pdS - 2*m*n*l*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.57 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement57){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][2].getMatrixElement(d[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(sqrt(3.)*n*n*l*V_pdS + l*(1 - 2*n*n)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.58 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement58){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][0].getMatrixElement(d[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(3*l*l*m*n*V_ddS + m*n*(1 - 4*l*l)*V_ddP + m*n*(l*l - 1)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.59 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement59){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][1].getMatrixElement(d[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(3*n*n*l*m*V_ddS + l*m*(1 - 4*n*n)*V_ddP + l*m*(n*n - 1)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.60 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement60){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][2].getMatrixElement(d[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), E_d, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(3*n*n*l*l*V_ddS + (n*n + l*l - 4*n*n*l*l)*V_ddP + (m*m + n*n*l*l)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.61 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement61){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][3].getMatrixElement(d[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((3./2.)*n*l*(l*l - m*m)*V_ddS + n*l*(1 - 2*(l*l - m*m))*V_ddP - n*l*(1 - (l*l - m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.62 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement62){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][4].getMatrixElement(d[j][2]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*(l*n*(n*n - (l*l + m*m)/2.)*V_ddS + l*n*(l*l + m*m - n*n)*V_ddP - (l*n*(l*l + m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.63 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement63){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= s[i][0].getMatrixElement(d[j][3]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(sqrt(3.)/2.)*(l*l - m*m)*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.64 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement64){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][0].getMatrixElement(d[j][3]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-((sqrt(3.)/2.)*l*(l*l - m*m)*V_pdS + l*(1 - l*l + m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.65 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement65){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][1].getMatrixElement(d[j][3]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-((sqrt(3.)/2.)*m*(l*l - m*m)*V_pdS - m*(1 + l*l - m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.66 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement66){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][2].getMatrixElement(d[j][3]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-((sqrt(3.)/2.)*n*(l*l - m*m)*V_pdS - n*(l*l - m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.67 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement67){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][0].getMatrixElement(d[j][3]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((3./2.)*l*m*(l*l - m*m)*V_ddS + 2*l*m*(m*m - l*l)*V_ddP + (l*m*(l*l - m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.68 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement68){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][1].getMatrixElement(d[j][3]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((3./2.)*m*n*(l*l - m*m)*V_ddS - m*n*(1 + 2*(l*l - m*m))*V_ddP + m*n*(1 + (l*l - m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.69 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement69){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][2].getMatrixElement(d[j][3]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((3./2.)*n*l*(l*l - m*m)*V_ddS + n*l*(1 - 2*(l*l - m*m))*V_ddP - n*l*(1 - (l*l - m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.70 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement70){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][3].getMatrixElement(d[j][3]);
			if(i == j){
				EXPECT_NEAR(real(result), E_d, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					((3./4.)*pow(l*l - m*m, 2)*V_ddS + (l*l + m*m - pow(l*l - m*m, 2))*V_ddP + (n*n + pow(l*l - m*m, 2)/4.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.71 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement71){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][4].getMatrixElement(d[j][3]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*((l*l - m*m)*(n*n - (l*l + m*m)/2.)*V_ddS/2. + n*n*(m*m - l*l)*V_ddP + ((1 + n*n)*(l*l - m*m)/4.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.72 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement72){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= s[i][0].getMatrixElement(d[j][4]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(n*n - (l*l + m*m)/2.)*V_sdS*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.73 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement73){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][0].getMatrixElement(d[j][4]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(l*(n*n - (l*l + m*m)/2.)*V_pdS - sqrt(3.)*l*n*n*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.74 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement74){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][1].getMatrixElement(d[j][4]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(m*(n*n - (l*l + m*m)/2.)*V_pdS - sqrt(3.)*m*n*n*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.75 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement75){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= p[i][2].getMatrixElement(d[j][4]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					-(n*(n*n - (l*l + m*m)/2.)*V_pdS + sqrt(3.)*n*(l*l + m*m)*V_pdP)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.76 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement76){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][0].getMatrixElement(d[j][4]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*(l*m*(n*n - (l*l + m*m)/2.)*V_ddS - 2*l*m*n*n*V_ddP + (l*m*(1 + n*n)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.77 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement77){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][1].getMatrixElement(d[j][4]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*(m*n*(n*n - (l*l + m*m)/2.)*V_ddS + m*n*(l*l + m*m - n*n)*V_ddP - (m*n*(l*l + m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.78 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement78){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][2].getMatrixElement(d[j][4]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*(l*n*(n*n - (l*l + m*m)/2.)*V_ddS + l*n*(l*l + m*m - n*n)*V_ddP - (l*n*(l*l + m*m)/2.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.79 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement79){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][3].getMatrixElement(d[j][4]);
			if(i == j){
				EXPECT_NEAR(real(result), 0, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					sqrt(3.)*((l*l - m*m)*(n*n - (l*l + m*m)/2.)*V_ddS/2. + n*n*(m*m - l*l)*V_ddP + ((1 + n*n)*(l*l - m*m)/4.)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

//TBTKFeature StatesAndOperators.SlaterKosterState.getMatrixElement.80 2020-05-30
TEST_F(SlaterKosterStateTest, getMatrixElement80){
	for(unsigned int i = 0; i < 4; i++){
		for(unsigned int j = 0; j < 4; j++){
			std::complex<double> result
				= d[i][4].getMatrixElement(d[j][4]);
			if(i == j){
				EXPECT_NEAR(real(result), E_d, EPSILON_100);
			}
			else{
				double l = getL(positions[i], positions[j]);
				double m = getM(positions[i], positions[j]);
				double n = getN(positions[i], positions[j]);
				EXPECT_NEAR(
					real(result),
					(pow(n*n - (l*l + m*m)/2., 2)*V_ddS + 3*n*n*(l*l + m*m)*V_ddP + (3./4.)*pow(l*l + m*m, 2)*V_ddD)*radialFunction(positions[i], positions[j]),
					EPSILON_100
				);
			}
		}
	}
}

};
