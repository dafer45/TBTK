#include "TBTK/UnitHandler.h"

#include "gtest/gtest.h"

#include <vector>

namespace TBTK{

class UnitHandlerTest : public ::testing::Test{
protected:
	//Error margin.
	constexpr static const double EPSILON = 1e-6;

	//Source "The International System of Units (SI) 9th Edition. Bureau
	//International des Poids et Mesures. 2019."
	constexpr static const double h = 6.62607015*1e-34;	//m^2 kg / s
	constexpr static const double e = 1.602176634*1e-19;	//C
	constexpr static const double k_B = 1.380649*1e-23;	//J / K
	constexpr static const double c = 2.99792458*1e8;	//m / s
	constexpr static const double N_A = 6.02214076*1e23;	//pcs / mol

	//Source "The NIST reference on Constants, Units, and Uncertainty."
	//https://physics.nist.gov/cuu/Constants/index.html
	constexpr static const double m_e = 9.1093837015*1e-31;		//kg = J s^2 / m^2
	constexpr static const double m_p = 1.67262192369*1e-27;	//kg = J s^2 / m^2
	constexpr static const double mu_0 = 1.25663706212*1e-6;	//N / A^2 = J s^2 / m C^2
	constexpr static const double epsilon_0 = 8.8541878128*1e-12;	//F / m = C^2 / J m
	constexpr static const double a_0 = 5.29177210903*1e-11;	//m

	constexpr static const double hbar = 6.62607015*1e-34/(2*M_PI);	//m^2 kg / s
	constexpr static const double mu_B = e*hbar/(2*m_e);		//C m^2 / s
	constexpr static const double mu_n = e*hbar/(2*m_p);		//C m^2 / s

	constexpr static const double J_per_eV = e;
	constexpr static const double pcs_per_mol = N_A;

	void SetUp() override{
		UnitHandler::setScales({
			"1.1 kC",
			"1.2 pcs",
			"1.3 GeV",
			"1.4 Ao",
			"1.5 kK",
			"1.6 as"
		});
	}
};

//TBTKFeature Utilities.UnitHandler.getHbarBase.1 2019-12-09
TEST_F(UnitHandlerTest, getHbarBase1){
	//[hbar]_L = GeV as = J_per_Gev*s_per_as Js
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	EXPECT_NEAR(
		UnitHandler::getHbarB()*J_per_GeV*s_per_as,
		hbar,
		hbar*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getHbarNatural.1 2019-12-09
TEST_F(UnitHandlerTest, getHbarNatural1){
	//[hbar] = 1.3*1.6 GeV as = 1.3*1.6*J_per_Gev*s_per_as Js
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	EXPECT_NEAR(
		UnitHandler::getHbarN()*(1.3*J_per_GeV*1.6*s_per_as),
		hbar,
		hbar*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getK_BBase.1 2019-12-09
TEST_F(UnitHandlerTest, getK_BBase1){
	//[k_B] = GeV/kK = J_per_GeV/K_perkK J/K
	double J_per_GeV = J_per_eV*1e9;
	double K_per_kK = 1e3;
	EXPECT_NEAR(
		UnitHandler::getK_BB()*J_per_GeV/K_per_kK,
		k_B,
		k_B*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getK_BNatural.1 2019-12-09
TEST_F(UnitHandlerTest, getK_BNatural1){
	//[k_B] = 1.3/1.5 GeV/kK = 1.3/1.5 J_perGeV/K_per_kK J/K
	double J_per_GeV = J_per_eV*1e9;
	double K_per_kK = 1e3;
	EXPECT_NEAR(
		UnitHandler::getK_BN()*(1.3*J_per_GeV)/(1.5*K_per_kK),
		k_B,
		k_B*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getEBase.1 2019-12-09
TEST_F(UnitHandlerTest, getEBase1){
	//[k_B] = kC = C_per_kC C
	double C_per_kC = 1e3;
	EXPECT_NEAR(UnitHandler::getEB()*C_per_kC, e, e*EPSILON);
}

//TBTKFeature Utilities.UnitHandler.getENatural.1 2019-12-09
TEST_F(UnitHandlerTest, getENatural1){
	//[k_B] = 1.1 kC = 1. 1C_per_kC C
	double C_per_kC = 1e3;
	EXPECT_NEAR(UnitHandler::getEN()*1.1*C_per_kC, e, e*EPSILON);
}

//TBTKFeature Utilities.UnitHandler.getCBase.1 2019-12-09
TEST_F(UnitHandlerTest, getCBase1){
	//[k_B] = Ao/as = m_per_Ao/s_per_as m/s
	double m_per_Ao = 1e-10;
	double s_per_as = 1e-18;
	EXPECT_NEAR(UnitHandler::getCB()*m_per_Ao/s_per_as, c, c*EPSILON);
}

//TBTKFeature Utilities.UnitHandler.getCNatural.1 2019-12-09
TEST_F(UnitHandlerTest, getCNatural1){
	//[k_B] = 1.4/1.6 Ao/as = 1.4/1.6 m_per_Ao/s_per_as m/s
	double m_per_Ao = 1e-10;
	double s_per_as = 1e-18;
	EXPECT_NEAR(
		UnitHandler::getCN()*(1.4*m_per_Ao)/(1.6*s_per_as),
		c,
		c*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getN_ABase.1 2019-12-09
TEST_F(UnitHandlerTest, getN_ABase1){
	//[N_A] = pcs/mol
	EXPECT_NEAR(UnitHandler::getN_AB(), N_A, N_A*EPSILON);
}

//TBTKFeature Utilities.UnitHandler.getN_ANatural.1 2019-12-09
TEST_F(UnitHandlerTest, getN_ANatural1){
	//[N_A] = 1.2 pcs/mol
	EXPECT_NEAR(UnitHandler::getN_AN()*1.2, N_A, N_A*EPSILON);
}

//TBTKFeature Utilities.UnitHandler.getM_eBase.1 2019-12-09
TEST_F(UnitHandlerTest, getM_eBase1){
	//[m_e] = GeV as^2/Ao^2 = J_per_GeV*(s_per_as)^2/(m_per_Ao)^2 Js^2/m^2
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	double m_per_Ao = 1e-10;
	EXPECT_NEAR(
		UnitHandler::getM_eB()*J_per_GeV*s_per_as*s_per_as/(
			m_per_Ao*m_per_Ao
		),
		m_e,
		m_e*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getM_eNatural.1 2019-12-09
TEST_F(UnitHandlerTest, getM_eNatural1){
	//[m_e] = 1.3*1.6^2/1.4^2 GeV as^2/Ao^2
	// = 1.3*1.6^2/1.4^2 J_per_GeV*(s_per_as)^2/(m_per_Ao)^2 Js^2/m^2
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	double m_per_Ao = 1e-10;
	EXPECT_NEAR(
		UnitHandler::getM_eN()*(1.3*J_per_GeV)*(1.6*s_per_as)*(
			1.6*s_per_as
		)/((1.4*m_per_Ao)*(1.4*m_per_Ao)),
		m_e,
		m_e*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getM_pBase.1 2019-12-09
TEST_F(UnitHandlerTest, getM_pBase1){
	//[m_p] = GeV as^2/Ao^2 = J_per_GeV*(s_per_as)^2/(m_per_Ao)^2 Js^2/m^2
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	double m_per_Ao = 1e-10;
	EXPECT_NEAR(
		UnitHandler::getM_pB()*J_per_GeV*s_per_as*s_per_as/(
			m_per_Ao*m_per_Ao
		),
		m_p,
		m_p*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getM_pNatural.1 2019-12-09
TEST_F(UnitHandlerTest, getM_pNatural1){
	//[m_p] = 1.3*1.6^2/1.4^2 GeV as^2/Ao^2
	// = 1.3*1.6^2/1.4^2 J_per_GeV*(s_per_as)^2/(m_per_Ao)^2 Js^2/m^2
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	double m_per_Ao = 1e-10;
	EXPECT_NEAR(
		UnitHandler::getM_pN()*(1.3*J_per_GeV)*(1.6*s_per_as)*(
			1.6*s_per_as
		)/((1.4*m_per_Ao)*(1.4*m_per_Ao)),
		m_p,
		m_p*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getMu_BBase.1 2019-12-09
TEST_F(UnitHandlerTest, getMu_BBase1){
	//[mu_B] = kC Ao^2/as = C_per_kC*(m_per_Ao)^2/s_per_as C m^2/s
	double C_per_kC = 1e3;
	double m_per_Ao = 1e-10;
	double s_per_as = 1e-18;
	EXPECT_NEAR(
		UnitHandler::getMu_BB()*C_per_kC*m_per_Ao*m_per_Ao/s_per_as,
		mu_B,
		mu_B*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getMu_BNatural.1 2019-12-09
TEST_F(UnitHandlerTest, getMu_BNatural1){
	//[mu_B] = kC Ao^2/as = C_per_kC*(m_per_Ao)^2/s_per_as C m^2/s
	double C_per_kC = 1e3;
	double m_per_Ao = 1e-10;
	double s_per_as = 1e-18;
	EXPECT_NEAR(
		UnitHandler::getMu_BN()*(1.1*C_per_kC)*(1.4*m_per_Ao)*(
			1.4*m_per_Ao
		)/(1.6*s_per_as),
		mu_B,
		mu_B*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getMu_nBase.1 2019-12-09
TEST_F(UnitHandlerTest, getMu_nBase1){
	//[mu_n] = kC Ao^2/as = C_per_kC*(m_per_Ao)^2/s_per_as C m^2/s
	double C_per_kC = 1e3;
	double m_per_Ao = 1e-10;
	double s_per_as = 1e-18;
	EXPECT_NEAR(
		UnitHandler::getMu_nB()*C_per_kC*m_per_Ao*m_per_Ao/s_per_as,
		mu_n,
		mu_n*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getMu_nNatural.1 2019-12-09
TEST_F(UnitHandlerTest, getMu_nNatural1){
	//[mu_n] = kC Ao^2/as = C_per_kC*(m_per_Ao)^2/s_per_as C m^2/s
	double C_per_kC = 1e3;
	double m_per_Ao = 1e-10;
	double s_per_as = 1e-18;
	EXPECT_NEAR(
		UnitHandler::getMu_nN()*(1.1*C_per_kC)*(1.4*m_per_Ao)*(
			1.4*m_per_Ao
		)/(1.6*s_per_as),
		mu_n,
		mu_n*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getMu_0Base.1 2019-12-09
TEST_F(UnitHandlerTest, getMu_0Base1){
	//[mu_0] = GeV as^2/Ao kC^2
	//= J_per_GeV*s_per_as^2/(m_per_ao*C_per_kC^2) J s^2 / m C^2
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	double m_per_Ao = 1e-10;
	double C_per_kC = 1e3;
	EXPECT_NEAR(
		UnitHandler::getMu_0B()*J_per_GeV*s_per_as*s_per_as/(m_per_Ao*C_per_kC*C_per_kC),
		mu_0,
		mu_0*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getMu_0Natural.1 2019-12-09
TEST_F(UnitHandlerTest, getMu_0Natural1){
	//[mu_0] = 1.3*1.6^2/(1.4*1.1^2) GeV as^2/Ao kC^2
	// = 1.3*1.6^2/(1.4*1.1^2) J_per_GeV*s_per_as^2/(m_per_ao*C_per_kC^2)
	// J s^2 / m C^2
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	double m_per_Ao = 1e-10;
	double C_per_kC = 1e3;
	EXPECT_NEAR(
		UnitHandler::getMu_0N()*(1.3*J_per_GeV)*(1.6*s_per_as)*(
			1.6*s_per_as
		)/((1.4*m_per_Ao)*(1.1*C_per_kC)*(1.1*C_per_kC)),
		mu_0,
		mu_0*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getEpsilon_0Base.1 2019-12-09
TEST_F(UnitHandlerTest, getEpsilon_0Base1){
	//[epsilon_0] = kC^2/GeV Ao = C_per_kC^2/(J_per_GeV*m_per_Ao) C^2/J m
	double C_per_kC = 1e3;
	double J_per_GeV = J_per_eV*1e9;
	double m_per_Ao = 1e-10;
	EXPECT_NEAR(
		UnitHandler::getEpsilon_0B()*C_per_kC*C_per_kC/(
			J_per_GeV*m_per_Ao
		),
		epsilon_0,
		epsilon_0*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getEpsilon_0Natural.1 2019-12-09
TEST_F(UnitHandlerTest, getEpsilon_0Natural1){
	//[epsilon_0] = 1.1^2/(1.3*1.4) kC^2/GeV Ao
	// = 1.1^2/(1.3*1.4) C_per_kC^2/(J_per_GeV*m_per_Ao) C^2/J m
	double C_per_kC = 1e3;
	double J_per_GeV = J_per_eV*1e9;
	double m_per_Ao = 1e-10;
	EXPECT_NEAR(
		UnitHandler::getEpsilon_0N()*(1.1*C_per_kC)*(1.1*C_per_kC)/(
			(1.3*J_per_GeV)*(1.4*m_per_Ao)
		),
		epsilon_0,
		epsilon_0*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getA_0Base.1 2019-12-09
TEST_F(UnitHandlerTest, getA_0Base1){
	//[a_0] = Ao = m_per_Ao m
	double m_per_Ao = 1e-10;
	EXPECT_NEAR(
		UnitHandler::getA_0B()*m_per_Ao,
		a_0,
		a_0*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.getA_0Natural.1 2019-12-09
TEST_F(UnitHandlerTest, getA_0Natural1){
	//[a_0] = 1.4 Ao = 1.4 m_per_Ao m
	double m_per_Ao = 1e-10;
	EXPECT_NEAR(
		UnitHandler::getA_0N()*1.4*m_per_Ao,
		a_0,
		a_0*EPSILON
	);
}

//TBTKFeature Utilities.UnitHandler.convertTemperatureNaturalToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertTemperatureNaturalToBase1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertTemperatureNtB(10), 10*1.5);
}

//TBTKFeature Utilities.UnitHandler.convertTimeNaturalToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertTimeNaturalToBase1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertTimeNtB(10), 10*1.6);
}

//TBTKFeature Utilities.UnitHandler.convertLengthNaturalToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertLengthNaturalToBase1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertLengthNtB(10), 10*1.4);
}

//TBTKFeature Utilities.UnitHandler.convertEnergyNaturalToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertEnergyNaturalToBase1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertEnergyNtB(10), 10*1.3);
}

//TBTKFeature Utilities.UnitHandler.convertChargeNaturalToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertChargeNaturalToBase1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertChargeNtB(10), 10*1.1);
}

//TBTKFeature Utilities.UnitHandler.convertCountNaturalToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertCountNaturalToBase1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertCountNtB(10), 10*1.2);
}

//TBTKFeature Utilities.UnitHandler.convertTemperatureBaseToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertTemperatureBaseToNatural1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertTemperatureBtN(10), 10/1.5);
}

//TBTKFeature Utilities.UnitHandler.convertTimeBaseToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertTimeBaseToNatural1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertTimeBtN(10), 10/1.6);
}

//TBTKFeature Utilities.UnitHandler.convertLengthBaseToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertLengthBaseToNatural1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertLengthBtN(10), 10/1.4);
}

//TBTKFeature Utilities.UnitHandler.convertEnergyBaseToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertEnergyBaseToNatural1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertEnergyBtN(10), 10/1.3);
}

//TBTKFeature Utilities.UnitHandler.convertChargeBaseToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertChargeBaseToNatural1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertChargeBtN(10), 10/1.1);
}

//TBTKFeature Utilities.UnitHandler.convertCountBaseToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertCountBaseToNatural1){
	EXPECT_DOUBLE_EQ(UnitHandler::convertCountBtN(10), 10/1.2);
}

//TBTKFeature Utilities.UnitHandler.convertTemperatureArbitraryToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertTemperatureArbitraryToBase1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertTemperatureAtB(
			10,
			UnitHandler::TemperatureUnit::uK
		),
		10*1e-9
	);
}

//TBTKFeature Utilities.UnitHandler.convertTimeArbitraryToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertTimeArbitraryToBase1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertTimeAtB(
			10,
			UnitHandler::TimeUnit::ns
		),
		10*1e9
	);
}

//TBTKFeature Utilities.UnitHandler.convertLengthArbitraryToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertLengthArbitraryToBase1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertLengthAtB(
			10,
			UnitHandler::LengthUnit::mm
		),
		10*1e7
	);
}

//TBTKFeature Utilities.UnitHandler.convertEnergyArbitraryToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertEnergyArbitraryToBase1){
	double eV_per_J = 1/J_per_eV;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertEnergyAtB(
			10,
			UnitHandler::EnergyUnit::J
		),
		10*eV_per_J*1e-9
	);
}

//TBTKFeature Utilities.UnitHandler.convertChargeArbitraryToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertChargeArbitraryToBase1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertChargeAtB(
			10,
			UnitHandler::ChargeUnit::uC
		),
		10*1e-9
	);
}

//TBTKFeature Utilities.UnitHandler.convertCountArbitraryToBase.1 2019-12-10
TEST_F(UnitHandlerTest, convertCountArbitraryToBase1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertCountAtB(
			10,
			UnitHandler::CountUnit::mol
		),
		10*N_A
	);
}

//TBTKFeature Utilities.UnitHandler.convertTemperatureBaseToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertTemperatureBaseToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertTemperatureBtA(
			10,
			UnitHandler::TemperatureUnit::uK
		),
		10/1e-9
	);
}

//TBTKFeature Utilities.UnitHandler.convertTimeBaseToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertTimeBaseToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertTimeBtA(
			10,
			UnitHandler::TimeUnit::ns
		),
		10/1e9
	);
}

//TBTKFeature Utilities.UnitHandler.convertLengthBaseToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertLengthBaseToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertLengthBtA(
			10,
			UnitHandler::LengthUnit::mm
		),
		10/1e7
	);
}

//TBTKFeature Utilities.UnitHandler.convertEnergyBaseToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertEnergyBaseToArbitrary1){
	double eV_per_J = 1/J_per_eV;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertEnergyBtA(
			10,
			UnitHandler::EnergyUnit::J
		),
		10/(eV_per_J*1e-9)
	);
}

//TBTKFeature Utilities.UnitHandler.convertChargeBaseToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertChargeBaseToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertChargeBtA(
			10,
			UnitHandler::ChargeUnit::uC
		),
		10/1e-9
	);
}

//TBTKFeature Utilities.UnitHandler.convertCountBaseToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertCountBaseToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertCountBtA(
			10,
			UnitHandler::CountUnit::mol
		),
		10/N_A
	);
}

//TBTKFeature Utilities.UnitHandler.convertTemperatureArbitraryToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertTemperatureArbitraryToNatural1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertTemperatureAtN(
			10,
			UnitHandler::TemperatureUnit::uK
		),
		10*1e-9/1.5
	);
}

//TBTKFeature Utilities.UnitHandler.convertTimeArbitraryToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertTimeArbitraryToNatural1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertTimeAtN(
			10,
			UnitHandler::TimeUnit::ns
		),
		10*1e9/1.6
	);
}

//TBTKFeature Utilities.UnitHandler.convertLengthArbitraryToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertLengthArbitraryToNatural1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertLengthAtN(
			10,
			UnitHandler::LengthUnit::mm
		),
		10*1e7/1.4
	);
}

//TBTKFeature Utilities.UnitHandler.convertEnergyArbitraryToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertEnergyArbitraryToNatural1){
	double eV_per_J = 1/J_per_eV;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertEnergyAtN(
			10,
			UnitHandler::EnergyUnit::J
		),
		10*eV_per_J*1e-9/1.3
	);
}

//TBTKFeature Utilities.UnitHandler.convertChargeArbitraryToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertChargeArbitraryToNatural1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertChargeAtN(
			10,
			UnitHandler::ChargeUnit::uC
		),
		10*1e-9/1.1
	);
}

//TBTKFeature Utilities.UnitHandler.convertCountArbitraryToNatural.1 2019-12-10
TEST_F(UnitHandlerTest, convertCountArbitraryToNatural1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertCountAtN(
			10,
			UnitHandler::CountUnit::mol
		),
		10*N_A/1.2
	);
}

//TBTKFeature Utilities.UnitHandler.convertTemperatureNaturalToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertTemperatureNaturalToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertTemperatureNtA(
			10,
			UnitHandler::TemperatureUnit::uK
		),
		1.5*10/1e-9
	);
}

//TBTKFeature Utilities.UnitHandler.convertTimeNaturalToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertTimeNaturalToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertTimeNtA(
			10,
			UnitHandler::TimeUnit::ns
		),
		1.6*10/1e9
	);
}

//TBTKFeature Utilities.UnitHandler.convertLengthNaturalToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertLengthNaturalToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertLengthNtA(
			10,
			UnitHandler::LengthUnit::mm
		),
		1.4*10/1e7
	);
}

//TBTKFeature Utilities.UnitHandler.convertEnergyNaturalToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertEnergyNaturalToArbitrary1){
	double eV_per_J = 1/J_per_eV;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertEnergyNtA(
			10,
			UnitHandler::EnergyUnit::J
		),
		1.3*10/(eV_per_J*1e-9)
	);
}

//TBTKFeature Utilities.UnitHandler.convertChargeNaturalToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertChargeNaturalToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertChargeNtA(
			10,
			UnitHandler::ChargeUnit::uC
		),
		1.1*10/1e-9
	);
}

//TBTKFeature Utilities.UnitHandler.convertCountNaturalToArbitrary.1 2019-12-10
TEST_F(UnitHandlerTest, convertCountNaturalToArbitrary1){
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertCountNtA(
			10,
			UnitHandler::CountUnit::mol
		),
		1.2*10/N_A
	);
}

//TBTKFeature Utilities.UnitHandler.convertMassDerivedToBase.1 2019-12-11
TEST_F(UnitHandlerTest, convertMassDerivedToBase1){
	//[mass] = ug = 10^-9 J s^2/m^2
	// = 10^-9 GeV_per_J*as_per_s^2/Ao_per_m^2 GeV as^2/Ao^2
	double GeV_per_J = 1e-9/J_per_eV;
	double as_per_s = 1e18;
	double Ao_per_m = 1e10;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertMassDtB(
			10,
			UnitHandler::MassUnit::ug
		),
		10*1e-9*GeV_per_J*as_per_s*as_per_s/(Ao_per_m*Ao_per_m)
	);
}

//TBTKFeature Utilities.UnitHandler.convertMassBaseToDerived.1 2019-12-11
TEST_F(UnitHandlerTest, convertMassBaseToDerived1){
	//[mass] = GeV as^2/Ao^2 = J_per_GeV*s_per_as^2/m_per_Ao^2 J s^2/m^2
	// = J_per_GeV*s_per_as^2/m_per_Ao^2 kg
	// = 10^9 J_per_GeV*s_per_as^2/m_per_Ao^2 ug
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	double m_per_Ao = 1e-10;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertMassBtD(
			10,
			UnitHandler::MassUnit::ug
		),
		10*1e9*J_per_GeV*s_per_as*s_per_as/(m_per_Ao*m_per_Ao)
	);
}

//TBTKFeature Utilities.UnitHandler.convertMassDerivedToNatural.1 2019-12-11
TEST_F(UnitHandlerTest, convertMassDerivedToNatural1){
	//[mass] = ug = 10^-9 J s^2/m^2
	// = 10^-9/(1.3*1.6^2/1.4^2) GeV_per_J*as_per_s^2/Ao_per_m^2
	// (1.3*1.6^2/1.4^2 GeV as^2/Ao^2)
	double GeV_per_J = 1e-9/J_per_eV;
	double as_per_s = 1e18;
	double Ao_per_m = 1e10;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertMassDtN(
			10,
			UnitHandler::MassUnit::ug
		),
		10*1e-9/(1.3*1.6*1.6/(1.4*1.4))*GeV_per_J*as_per_s*as_per_s/(Ao_per_m*Ao_per_m)
	);
}

//TBTKFeature Utilities.UnitHandler.convertMassNaturalToDerived.1 2019-12-11
TEST_F(UnitHandlerTest, convertMassNaturalToDerived1){
	//[mass] = 1.3*1.6^2/1.4^2 GeV as^2/Ao^2
	// = 1.3*1.6^2/1.4^2 J_per_GeV*s_per_as^2/m_per_Ao^2 J s^2/m^2
	// = 1.3*1.6^2/1.4^2 J_per_GeV*s_per_as^2/m_per_Ao^2 kg
	// = 10^9*1.3*1.6^2/1.4^2 J_per_GeV*s_per_as^2/m_per_Ao^2 ug
	double J_per_GeV = J_per_eV*1e9;
	double s_per_as = 1e-18;
	double m_per_Ao = 1e-10;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertMassNtD(
			10,
			UnitHandler::MassUnit::ug
		),
		10*1e9*(1.3*1.6*1.6/(1.4*1.4))*J_per_GeV*s_per_as*s_per_as/(
			m_per_Ao*m_per_Ao
		)
	);
}

//TBTKFeature Utilities.UnitHandler.convertMagneticFieldDerivedToBase.1 2019-12-11
TEST_F(UnitHandlerTest, convertMagneticFieldDerivedToBase1){
	//[magnetic field] = uT = 10^-6 Js/Cm^2
	// = 10^-6 GeV_per_J*as_per_s/(kC_per_C*Ao_per_m^2) GeV as/kC Ao^2
	double GeV_per_J = 1e-9/J_per_eV;
	double as_per_s = 1e18;
	double kC_per_C = 1e-3;
	double Ao_per_m = 1e10;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertMagneticFieldDtB(
			10,
			UnitHandler::MagneticFieldUnit::uT
		),
		10*1e-6*GeV_per_J*as_per_s/(kC_per_C*Ao_per_m*Ao_per_m)
	);
}

//TBTKFeature Utilities.UnitHandler.convertMagneticFieldBaseToDerived.1 2019-12-11
TEST_F(UnitHandlerTest, convertMagneticFieldBaseToDerived1){
	//[magnetic field] = GeV as/kC Ao^2
	// = J_per_GeV*s_per_as/(C_per_kC*m_per_Ao^2) Js/Cm^2
	// = 10^6 J_per_GeV*s_per_as/(C_per_kC*m_per_Ao^2) uT
	double J_per_GeV = 1e9*J_per_eV;
	double s_per_as = 1e-18;
	double C_per_kC = 1e3;
	double m_per_Ao = 1e-10;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertMagneticFieldBtD(
			10,
			UnitHandler::MagneticFieldUnit::uT
		),
		10*1e6*J_per_GeV*s_per_as/(C_per_kC*m_per_Ao*m_per_Ao)
	);
}

//TBTKFeature Utilities.UnitHandler.convertMagneticFieldDerivedToNatural.1 2019-12-11
TEST_F(UnitHandlerTest, convertMagneticFieldDerivedToNatural1){
	//[magnetic field] = uT = 10^-6 Js/Cm^2
	// = 10^-6/(1.3*1.6/(1.1*1.4^2))
	// GeV_per_J*as_per_s/(kC_per_C*Ao_per_m^2)
	// (1.3*1.6/(1.1*1.4^2) GeV as/kC Ao^2)
	double GeV_per_J = 1e-9/J_per_eV;
	double as_per_s = 1e18;
	double kC_per_C = 1e-3;
	double Ao_per_m = 1e10;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertMagneticFieldDtN(
			10,
			UnitHandler::MagneticFieldUnit::uT
		),
		10*1e-6/(1.3*1.6/(1.1*1.4*1.4))*GeV_per_J*as_per_s/(
			kC_per_C*Ao_per_m*Ao_per_m
		)
	);
}

//TBTKFeature Utilities.UnitHandler.convertMagneticFieldNaturalToDerived.1 2019-12-11
TEST_F(UnitHandlerTest, convertMagneticFieldNaturalToDerived1){
	//[magnetic field] = 1.3*1.6/(1.1*1.4^2) GeV as/kC Ao^2
	// = 1.3*1.6/(1.1*1.4^2) J_per_GeV*s_per_as/(C_per_kC*m_per_Ao^2) Js/Cm^2
	// = 10^6*1.3*1.6/(1.1*1.4^2) J_per_GeV*s_per_as/(C_per_kC*m_per_Ao^2) uT
	double J_per_GeV = 1e9*J_per_eV;
	double s_per_as = 1e-18;
	double C_per_kC = 1e3;
	double m_per_Ao = 1e-10;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertMagneticFieldNtD(
			10,
			UnitHandler::MagneticFieldUnit::uT
		),
		10*1e6*(1.3*1.6/(1.1*1.4*1.4))*J_per_GeV*s_per_as/(
			C_per_kC*m_per_Ao*m_per_Ao
		)
	);
}

//TBTKFeature Utilities.UnitHandler.convertVoltageDerivedToBase.1 2019-12-11
TEST_F(UnitHandlerTest, convertVoltageDerivedToBase1){
	//[voltage] = uV = 10^-6 J/C = 10^-6 GeV_per_J/kC_per_C GeV/kC
	double GeV_per_J = 1e-9/J_per_eV;
	double kC_per_C = 1e-3;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertVoltageDtB(
			10,
			UnitHandler::VoltageUnit::uV
		),
		10*1e-6*GeV_per_J/kC_per_C
	);
}

//TBTKFeature Utilities.UnitHandler.convertVoltageBaseToDerived.1 2019-12-11
TEST_F(UnitHandlerTest, convertVoltageBaseToDerived1){
	//[voltage] = GeV/kC = 10^6 J_per_GeV/C_per_kC uV
	double J_per_GeV = 1e9*J_per_eV;
	double C_per_kC = 1e3;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertVoltageBtD(
			10,
			UnitHandler::VoltageUnit::uV
		),
		10*1e6*J_per_GeV/C_per_kC
	);
}

//TBTKFeature Utilities.UnitHandler.convertVoltageDerivedToNatural.1 2019-12-11
TEST_F(UnitHandlerTest, convertVoltageDerivedToNatural1){
	//[voltage] = uV = 10^-6 J/C
	// = 10^-6/(1.3/1.1) GeV_per_J/kC_per_C (1.3/1.1 GeV/kC)
	double GeV_per_J = 1e-9/J_per_eV;
	double kC_per_C = 1e-3;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertVoltageDtN(
			10,
			UnitHandler::VoltageUnit::uV
		),
		10*1e-6/(1.3/1.1)*GeV_per_J/kC_per_C
	);
}

//TBTKFeature Utilities.UnitHandler.convertVoltageNaturalToDerived.1 2019-12-11
TEST_F(UnitHandlerTest, convertVoltageNaturalToDerived1){
	//[voltage] = GeV/kC = 10^6 J_per_GeV/C_per_kC uV
	double J_per_GeV = 1e9*J_per_eV;
	double C_per_kC = 1e3;
	EXPECT_DOUBLE_EQ(
		UnitHandler::convertVoltageNtD(
			10,
			UnitHandler::VoltageUnit::uV
		),
		10*1e6*(1.3/1.1)*J_per_GeV/C_per_kC
	);
}

};
