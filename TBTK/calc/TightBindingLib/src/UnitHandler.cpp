/** @file UnitHandler.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/UnitHandler.h"
#include <sstream>
#include <iostream>

using namespace std;

namespace TBTK{

double UnitHandler::hbar	= HBAR;
double UnitHandler::k_b		= K_B;
double UnitHandler::e		= E;
double UnitHandler::c		= C;
double UnitHandler::m_e		= M_E;
double UnitHandler::m_p		= M_P;
double UnitHandler::mu_b	= MU_B;
double UnitHandler::mu_n	= MU_N;

UnitHandler::TemperatureUnit 	UnitHandler::temperatureUnit	= UnitHandler::TemperatureUnit::K;
UnitHandler::TimeUnit 		UnitHandler::timeUnit		= UnitHandler::TimeUnit::s;
UnitHandler::LengthUnit		UnitHandler::lengthUnit		= UnitHandler::LengthUnit::m;
UnitHandler::EnergyUnit		UnitHandler::energyUnit		= UnitHandler::EnergyUnit::eV;
UnitHandler::ChargeUnit		UnitHandler::chargeUnit		= UnitHandler::ChargeUnit::C;

double UnitHandler::temperatureScale	= 1.;
double UnitHandler::timeScale		= 1.;
double UnitHandler::lengthScale		= 1.;
double UnitHandler::energyScale		= 1.;
double UnitHandler::chargeScale		= 1.;

void UnitHandler::setTemperatureUnit(TemperatureUnit unit){
	double oldConversionFactor = getTemperatureConversionFactor();
	temperatureUnit = unit;
	double newConversionFactor = getTemperatureConversionFactor();
	temperatureScale *= newConversionFactor/oldConversionFactor;

	updateK_b();
}

void UnitHandler::setTimeUnit(TimeUnit unit){
	double oldConversionFactor = getTimeConversionFactor();	
	timeUnit = unit;
	double newConversionFactor = getTimeConversionFactor();
	timeScale *= newConversionFactor/oldConversionFactor;

	updateHbar();
	updateC();
	updateM_e();
	updateM_p();
	updateMu_b();
	updateMu_n();
}

void UnitHandler::setLengthUnit(LengthUnit unit){
	double oldConversionFactor = getLengthConversionFactor();
	lengthUnit = unit;
	double newConversionFactor = getLengthConversionFactor();
	lengthScale *= newConversionFactor/oldConversionFactor;

	updateC();
	updateM_e();
	updateM_p();
	updateMu_b();
	updateMu_n();
}

void UnitHandler::setEnergyUnit(EnergyUnit unit){
	double oldConversionFactor = getEnergyConversionFactor();
	energyUnit = unit;
	double newConversionFactor = getEnergyConversionFactor();
	energyScale *= newConversionFactor/oldConversionFactor;

	updateHbar();
	updateK_b();
	updateM_e();
	updateM_p();
}

void UnitHandler::setChargeUnit(ChargeUnit unit){
	double oldConversionFactor = getChargeConversionFactor();
	chargeUnit = unit;
	double newConversionFactor = getChargeConversionFactor();
	chargeScale *= newConversionFactor/oldConversionFactor;

	updateE();
	updateMu_b();
	updateMu_n();
}

void UnitHandler::setTemperatureScale(double scale){
	temperatureScale = scale;
}

void UnitHandler::setTimeScale(double scale){
	timeScale = scale;
}

void UnitHandler::setLengthScale(double scale){
	lengthScale = scale;
}

void UnitHandler::setEnergyScale(double scale){
	energyScale = scale;
}

void UnitHandler::setChargeScale(double scale){
	chargeScale = scale;
}

double UnitHandler::convertMassDtB(double mass, MassUnit unit){
	double massInDefaultBaseUnits = mass/getMassConversionFactor(unit);
	double cfE = getEnergyConversionFactor();
	double cfT = getTimeConversionFactor();
	double cfL = getLengthConversionFactor();
	return massInDefaultBaseUnits*cfE*cfT*cfT/(cfL*cfL);
}

double UnitHandler::convertMassBtD(double mass, MassUnit unit){
	double cfE = getEnergyConversionFactor();
	double cfT = getTimeConversionFactor();
	double cfL = getLengthConversionFactor();
	double massInDefaultBaseUnits = mass*cfL*cfL/(cfE*cfT*cfT);
	return massInDefaultBaseUnits*getMassConversionFactor(unit);
}

double UnitHandler::convertMassDtN(double mass, MassUnit unit){
	double massInDefaultBaseUnits = mass/getMassConversionFactor(unit);
	double cfE = getEnergyConversionFactor()/energyScale;
	double cfT = getTimeConversionFactor()/timeScale;
	double cfL = getLengthConversionFactor()/lengthScale;
	return massInDefaultBaseUnits*cfE*cfT*cfT/(cfL*cfL);
}

double UnitHandler::convertMassNtD(double mass, MassUnit unit){
	double cfE = getEnergyConversionFactor()*energyScale;
	double cfT = getTimeConversionFactor()*timeScale;
	double cfL = getLengthConversionFactor()*lengthScale;
	double massInDefaultBaseUnits = mass*cfL*cfL/(cfE*cfT*cfT);
	return massInDefaultBaseUnits*getMassConversionFactor(unit);
}

double UnitHandler::convertMagneticFieldDtB(double field, MagneticFieldUnit unit){
	double magneticFieldInDefaultBaseUnits = field/getMagneticFieldConversionFactor(unit);
	double cfE = getEnergyConversionFactor();
	double cfT = getTimeConversionFactor();
	double cfC = getChargeConversionFactor();
	double cfL = getLengthConversionFactor();
	return magneticFieldInDefaultBaseUnits*cfE*cfT/(cfC*cfL*cfL);
}

double UnitHandler::convertMagneticFieldBtD(double field, MagneticFieldUnit unit){
	double cfE = getEnergyConversionFactor();
	double cfT = getTimeConversionFactor();
	double cfC = getChargeConversionFactor();
	double cfL = getLengthConversionFactor();
	double magneticFieldInDefaultBaseUnits = field*cfC*cfL*cfL/(cfE*cfT);
	return magneticFieldInDefaultBaseUnits*getMagneticFieldConversionFactor(unit);
}

double UnitHandler::convertMagneticFieldDtN(double field, MagneticFieldUnit unit){
	double magneticFieldInDefaultBaseUnits = field/getMagneticFieldConversionFactor(unit);
	double cfE = getEnergyConversionFactor()/energyScale;
	double cfT = getTimeConversionFactor()/timeScale;
	double cfC = getChargeConversionFactor()/chargeScale;
	double cfL = getLengthConversionFactor()/lengthScale;
	return magneticFieldInDefaultBaseUnits*cfE*cfT/(cfC*cfL*cfL);
}

double UnitHandler::convertMagneticFieldNtD(double field, MagneticFieldUnit unit){
	double cfE = getEnergyConversionFactor()/energyScale;
	double cfT = getTimeConversionFactor()/timeScale;
	double cfC = getChargeConversionFactor()/chargeScale;
	double cfL = getLengthConversionFactor()/lengthScale;
	double magneticFieldInDefaultBaseUnits = field*cfC*cfL*cfL/(cfE*cfT);
	return magneticFieldInDefaultBaseUnits*getMagneticFieldConversionFactor(unit);
}

string UnitHandler::getTemperatureUnitString(){
	switch(temperatureUnit){
		case TemperatureUnit::K:
			return "K";
		default:
			return "Unknown unit";
	};
}

string UnitHandler::getTimeUnitString(){
	switch(timeUnit){
		case TimeUnit::s:
			return "s";
		case TimeUnit::ms:
			return "ms";
		case TimeUnit::us:
			return "us";
		case TimeUnit::ns:
			return "ns";
		case TimeUnit::ps:
			return "ps";
		case TimeUnit::fs:
			return "fs";
		case TimeUnit::as:
			return "as";
		default:
			return "Unknown unit";
	};
}

string UnitHandler::getLengthUnitString(){
	switch(lengthUnit){
		case LengthUnit::m:
			return "m";
		case LengthUnit::mm:
			return "mm";
		case LengthUnit::um:
			return "um";
		case LengthUnit::nm:
			return "nm";
		case LengthUnit::pm:
			return "pm";
		case LengthUnit::fm:
			return "fm";
		case LengthUnit::am:
			return "am";
		case LengthUnit::Ao:
			return "Ao";
		default:
			return "Unknown unit";
	};
}

string UnitHandler::getEnergyUnitString(){
	switch(energyUnit){
		case EnergyUnit::GeV:
			return "GeV";
		case EnergyUnit::MeV:
			return "MeV";
		case EnergyUnit::keV:
			return "keV";
		case EnergyUnit::eV:
			return "eV";
		case EnergyUnit::meV:
			return "meV";
		case EnergyUnit::ueV:
			return "ueV";
		case EnergyUnit::J:
			return "J";
		default:
			return "Unknown unit";
	};
}

string UnitHandler::getChargeUnitString(){
	switch(chargeUnit){
		case ChargeUnit::C:
			return "C";
		default:
			return "Unknown unit";
	}
}

string UnitHandler::getHBARUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << getTimeUnitString();

	return ss.str();
}

string UnitHandler::getK_BUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << "/" << getTemperatureUnitString();

	return ss.str();
}

string UnitHandler::getEUnitString(){
	return getChargeUnitString();
}

string UnitHandler::getCUnitString(){
	stringstream ss;
	ss << getLengthUnitString() << getLengthUnitString() << "/" << getTimeUnitString() << getTimeUnitString();

	return ss.str();
}

string UnitHandler::getM_eUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << getTimeUnitString() << getTimeUnitString() << "/" << getLengthUnitString() << getLengthUnitString();

	return ss.str();
}

string UnitHandler::getM_pUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << getTimeUnitString() << getTimeUnitString() << "/" << getLengthUnitString() << getLengthUnitString();

	return ss.str();
}

string UnitHandler::getMu_bUnitString(){
	stringstream ss;
	ss << getChargeUnitString() << getLengthUnitString() << getLengthUnitString() << "/" << getTimeUnitString() << getTimeUnitString();

	return ss.str();
}

string UnitHandler::getMu_nUnitString(){
	stringstream ss;
	ss << getChargeUnitString() << getLengthUnitString() << getLengthUnitString() << "/" << getTimeUnitString() << getTimeUnitString();

	return ss.str();
}

void UnitHandler::updateHbar(){
	hbar = HBAR;
	hbar *= getEnergyConversionFactor();
	hbar *= getTimeConversionFactor();
}

void UnitHandler::updateK_b(){
	k_b = K_B;
	k_b *= getEnergyConversionFactor();
	k_b /= getTemperatureConversionFactor();
}

void UnitHandler::updateE(){
	e = E;
	e *= getChargeConversionFactor();
}

void UnitHandler::updateC(){
	c = C;
	c *= getLengthConversionFactor()*getLengthConversionFactor();
	c /= getTimeConversionFactor()*getTimeConversionFactor();
}

void UnitHandler::updateM_e(){
	m_e = M_E;
	m_e *= getEnergyConversionFactor();
	m_e *= getTimeConversionFactor()*getTimeConversionFactor();
	m_e /= getLengthConversionFactor()*getLengthConversionFactor();
}

void UnitHandler::updateM_p(){
	m_p = M_P;
	m_p *= getEnergyConversionFactor();
	m_p *= getTimeConversionFactor()*getTimeConversionFactor();
	m_p /= getLengthConversionFactor()*getLengthConversionFactor();
}

void UnitHandler::updateMu_b(){
	mu_b = MU_B;
	mu_b *= getChargeConversionFactor();
	mu_b *= getLengthConversionFactor()*getLengthConversionFactor();
	mu_b /= getTimeConversionFactor()*getTimeConversionFactor();
}

void UnitHandler::updateMu_n(){
	mu_n = MU_N;
	mu_n *= getChargeConversionFactor();
	mu_n *= getLengthConversionFactor()*getLengthConversionFactor();
	mu_n /= getTimeConversionFactor()*getTimeConversionFactor();
}

double UnitHandler::getTemperatureConversionFactor(){
	switch(temperatureUnit){
		case TemperatureUnit::K:	//Reference scale
			return 1.;
		default:	//Should never happen, hard error generated for quick bug detection
			cout << "Error in UnitHandler::getTemperatureConversionUnit(): Unknown unit - " << static_cast<int>(temperatureUnit);
			exit(1);
			return 0.;	//Never happens
	}
}

double UnitHandler::getTimeConversionFactor(){
	switch(timeUnit){
		case TimeUnit::s:	//Reference scale
			return 1.;
		case TimeUnit::ms:
			return 1e3;	//1e3 ms per second
		case TimeUnit::us:
			return 1e6;
		case TimeUnit::ns:
			return 1e9;
		case TimeUnit::ps:
			return 1e12;
		case TimeUnit::fs:
			return 1e15;
		case TimeUnit::as:
			return 1e18;
		default:	//Should never happen, hard error generated for quick bug detection
			cout << "Error in UnitHandler::getTimeConversionFactor(): Unknown unit - " << static_cast<int>(timeUnit);
			exit(1);
			return 0.;	//Never happens
	}
}

double UnitHandler::getLengthConversionFactor(){
	switch(lengthUnit){
		case LengthUnit::m:	//Reference scale
			return 1.;
		case LengthUnit::mm:	//1e3 mm per m
			return 1e3;
		case LengthUnit::um:
			return 1e6;
		case LengthUnit::nm:
			return 1e9;
		case LengthUnit::pm:
			return 1e12;
		case LengthUnit::fm:
			return 1e15;
		case LengthUnit::am:
			return 1e18;
		case LengthUnit::Ao:
			return 1e10;
		default:	//Should never happen, hard error generated for quick bug detection
			cout << "Error in UnitHandler::getLengthConversionFactor(): Unknown unit - " << static_cast<int>(lengthUnit);
			exit(1);
			return 0.;	//Never happens
	}
}

double UnitHandler::getEnergyConversionFactor(){
	switch(energyUnit){
		case EnergyUnit::GeV:	//1e-9 GeV per eV
			return 1e-9;
		case EnergyUnit::MeV:
			return 1e-6;
		case EnergyUnit::keV:
			return 1e-3;
		case EnergyUnit::eV:	//Reference scale
			return 1.;
		case EnergyUnit::meV:
			return 1e3;
		case EnergyUnit::ueV:
			return 1e6;
		case EnergyUnit::J:
			return J_per_eV;
		default:	//Should never happen, hard error generated for quick bug detection
			cout << "Error in UnitHandler::getEnergyConversionFactor(): Unknown unit - " << static_cast<int>(energyUnit);
			exit(1);
			return 0.;	//Never happens
	}
}

double UnitHandler::getChargeConversionFactor(){
	switch(chargeUnit){
		case ChargeUnit::C:
			return 1.;
		default:	//Should never happen, hard error generated for quick bug detection
			cout << "Error in UnitHandler::getChargeConversionFactor(): Unknown unit - " << static_cast<int>(chargeUnit);
			exit(1);
			return 0.;	//Never happens
	}
}

double UnitHandler::getMassConversionFactor(MassUnit unit){
	switch(unit){
		case MassUnit::kg:
			return kg_per_baseMass;
		case MassUnit::g:
			return kg_per_baseMass*1e3;
		case MassUnit::mg:
			return kg_per_baseMass*1e6;
		case MassUnit::ug:
			return kg_per_baseMass*1e9;
		case MassUnit::ng:
			return kg_per_baseMass*1e12;
		case MassUnit::u:
			return u_per_baseMass;
		default:	//Should never happen, hard error generated for quick bug detection
			cout << "Error in UnitHandler::getMassConversionFactor(): Unknown unit - " << static_cast<int>(unit);
			exit(1);
			return 0.;	//Never happens
	}
}

double UnitHandler::getMagneticFieldConversionFactor(MagneticFieldUnit unit){
	switch(unit){
		case MagneticFieldUnit::MT:
			return T_per_baseMagneticField*1e-6;
		case MagneticFieldUnit::kT:
			return T_per_baseMagneticField*1e-3;
		case MagneticFieldUnit::T:
			return T_per_baseMagneticField;
		case MagneticFieldUnit::mT:
			return T_per_baseMagneticField*1e3;
		case MagneticFieldUnit::uT:
			return T_per_baseMagneticField*1e6;
		case MagneticFieldUnit::nT:
			return T_per_baseMagneticField*1e9;
		case MagneticFieldUnit::GG:
			return T_per_baseMagneticField*1e-5;
		case MagneticFieldUnit::MG:
			return T_per_baseMagneticField*1e-2;
		case MagneticFieldUnit::kG:
			return T_per_baseMagneticField*10.;
		case MagneticFieldUnit::G:
			return T_per_baseMagneticField*1e4;	//10^4G = 1T
		case MagneticFieldUnit::mG:
			return T_per_baseMagneticField*1e7;
		case MagneticFieldUnit::uG:
			return T_per_baseMagneticField*1e10;
		default:	//Should never happen, hard error generated for quick bug detection
			cout << "Error in UnitHandler::getMagneticFieldConversionFactor(): Unknown unit - " << static_cast<int>(unit);
			exit(1);
			return 0.;	//Never happens
	}
}

};
