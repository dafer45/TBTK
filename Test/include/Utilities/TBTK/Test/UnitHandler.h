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

	constexpr static const double hbar = 6.62607015*1e-34/(2*M_PI);	//m^2 kg / s

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
	//[m_e] = GeV as^2/Ao^2
	// = (J_per_GeV*(s_per_as)^2/(m_per_Ao)^2) Js^2/m^2
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
	// = 1.3*1.6^2/1.4^2(J_per_GeV*(s_per_as)^2/(m_per_Ao)^2) Js^2/m^2
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
	//[m_p] = GeV as^2/Ao^2
	// = (J_per_GeV*(s_per_as)^2/(m_per_Ao)^2) Js^2/m^2
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
	// = 1.3*1.6^2/1.4^2(J_per_GeV*(s_per_as)^2/(m_per_Ao)^2) Js^2/m^2
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

};
