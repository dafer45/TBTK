/* Copyright 2016 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file UnitHandler.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/UnitHandler.h"

#include <iostream>
#include <sstream>

#ifdef M_E	//Avoid name clash with math.h macro M_E
	#define M_E_temp M_E
	#undef M_E
#endif

using namespace std;

namespace TBTK{

map<
	string,
	pair<double, vector<pair<string, int>>>
> UnitHandler::constantsDefaultUnits;
map<string, double> UnitHandler::constantsBaseUnits;

double UnitHandler::kg_per_baseMass;
double UnitHandler::baseMass_per_kg;
double UnitHandler::u_per_baseMass;
double UnitHandler::baseMass_per_u;
double UnitHandler::T_per_baseMagneticField;
double UnitHandler::baseMagneticField_per_T;
double UnitHandler::V_per_baseVoltage;
double UnitHandler::baseVoltage_per_V;

tuple<
	Quantity::Charge::Unit,
	Quantity::Count::Unit,
	Quantity::Energy::Unit,
	Quantity::Length::Unit,
	Quantity::Temperature::Unit,
	Quantity::Time::Unit
> UnitHandler::units = make_tuple(
	Quantity::Charge::Unit::C,
	Quantity::Count::Unit::pcs,
	Quantity::Energy::Unit::eV,
	Quantity::Length::Unit::m,
	Quantity::Temperature::Unit::K,
	Quantity::Time::Unit::s
);

tuple<double, double, double, double, double, double> UnitHandler::scales
	= make_tuple(1, 1, 1, 1, 1, 1);

UnitHandler::StaticConstructor  UnitHandler::staticConstructor;

double UnitHandler::getConstantBaseUnits(const std::string &name){
	return constantsBaseUnits.at(name);
}

double UnitHandler::getConstantNaturalUnits(const std::string &name){
	double value = getConstantBaseUnits(name);

	const vector<pair<string, int>> &units
		= constantsDefaultUnits.at(name).second;
	for(unsigned int n = 0; n < units.size(); n++){
		const string &unit = units[n].first;
		int exponent = units[n].second;
		for(int c = 0; c < exponent; c++){
			if(unit.compare("K") == 0){
				value /= getScale<Quantity::Temperature>();
			}
			else if(unit.compare("s") == 0){
				value /= getScale<Quantity::Time>();
			}
			else if(unit.compare("m") == 0){
				value /= getScale<Quantity::Length>();
			}
			else if(unit.compare("eV") == 0){
				value /= getScale<Quantity::Energy>();
			}
			else if(unit.compare("C") == 0){
				value /= getScale<Quantity::Charge>();
			}
			else if(unit.compare("pcs") == 0){
				value /= getScale<Quantity::Count>();
			}
			else{
				TBTKExit(
					"UnitHandler::getConstantNaturalUnits()",
					"Unknown default unit.",
					"This should never happen,"
					<< " contact the developer."
				);
			}
		}
		for(int c = 0; c < -exponent; c++){
			if(unit.compare("K") == 0){
				value *= getScale<Quantity::Temperature>();
			}
			else if(unit.compare("s") == 0){
				value *= getScale<Quantity::Time>();
			}
			else if(unit.compare("m") == 0){
				value *= getScale<Quantity::Length>();
			}
			else if(unit.compare("eV") == 0){
				value *= getScale<Quantity::Energy>();
			}
			else if(unit.compare("C") == 0){
				value *= getScale<Quantity::Charge>();
			}
			else if(unit.compare("pcs") == 0){
				value *= getScale<Quantity::Count>();
			}
			else{
				TBTKExit(
					"UnitHandler::getConstantNaturalUnits()",
					"Unknown default unit.",
					"This should never happen,"
					<< " contact the developer."
				);
			}
		}
	}

	return value;
}

/*double UnitHandler::convertMassDerivedToBase(double mass, MassUnit unit){
	double massInDefaultBaseUnits = mass/getMassConversionFactor(unit);
	double cfE = getConversionFactor<Quantity::Energy>();
	double cfT = getConversionFactor<Quantity::Time>();
	double cfL = getConversionFactor<Quantity::Length>();
	return massInDefaultBaseUnits*cfE*cfT*cfT/(cfL*cfL);
}

double UnitHandler::convertMassBaseToDerived(double mass, MassUnit unit){
	double cfE = getConversionFactor<Quantity::Energy>();
	double cfT = getConversionFactor<Quantity::Time>();
	double cfL = getConversionFactor<Quantity::Length>();
	double massInDefaultBaseUnits = mass*cfL*cfL/(cfE*cfT*cfT);
	return massInDefaultBaseUnits*getMassConversionFactor(unit);
}

double UnitHandler::convertMassDerivedToNatural(double mass, MassUnit unit){
	double massInDefaultBaseUnits = mass/getMassConversionFactor(unit);
	double cfE = getConversionFactor<Quantity::Energy>(
		)/getScale<Quantity::Energy>();
	double cfT = getConversionFactor<Quantity::Time>(
		)/getScale<Quantity::Time>();
	double cfL = getConversionFactor<Quantity::Length>(
		)/getScale<Quantity::Length>();
	return massInDefaultBaseUnits*cfE*cfT*cfT/(cfL*cfL);
}

double UnitHandler::convertMassNaturalToDerived(double mass, MassUnit unit){
	double cfE = getConversionFactor<Quantity::Energy>(
		)/getScale<Quantity::Energy>();
	double cfT = getConversionFactor<Quantity::Time>(
		)/getScale<Quantity::Time>();
	double cfL = getConversionFactor<Quantity::Length>(
		)/getScale<Quantity::Length>();
	double massInDefaultBaseUnits = mass*cfL*cfL/(cfE*cfT*cfT);
	return massInDefaultBaseUnits*getMassConversionFactor(unit);
}

double UnitHandler::convertMagneticFieldDerivedToBase(
	double field,
	MagneticFieldUnit unit
){
	double magneticFieldInDefaultBaseUnits = field/getMagneticFieldConversionFactor(unit);
	double cfE = getConversionFactor<Quantity::Energy>();
	double cfT = getConversionFactor<Quantity::Time>();
	double cfC = getConversionFactor<Quantity::Charge>();
	double cfL = getConversionFactor<Quantity::Length>();
	return magneticFieldInDefaultBaseUnits*cfE*cfT/(cfC*cfL*cfL);
}

double UnitHandler::convertMagneticFieldBaseToDerived(
	double field,
	MagneticFieldUnit unit
){
	double cfE = getConversionFactor<Quantity::Energy>();
	double cfT = getConversionFactor<Quantity::Time>();
	double cfC = getConversionFactor<Quantity::Charge>();
	double cfL = getConversionFactor<Quantity::Length>();
	double magneticFieldInDefaultBaseUnits = field*cfC*cfL*cfL/(cfE*cfT);
	return magneticFieldInDefaultBaseUnits*getMagneticFieldConversionFactor(unit);
}

double UnitHandler::convertMagneticFieldDerivedToNatural(
	double field,
	MagneticFieldUnit unit
){
	double magneticFieldInDefaultBaseUnits = field/getMagneticFieldConversionFactor(unit);
	double cfE = getConversionFactor<Quantity::Energy>(
		)/getScale<Quantity::Energy>();
	double cfT = getConversionFactor<Quantity::Time>(
		)/getScale<Quantity::Time>();
	double cfC = getConversionFactor<Quantity::Charge>(
		)/getScale<Quantity::Charge>();
	double cfL = getConversionFactor<Quantity::Length>(
		)/getScale<Quantity::Length>();
	return magneticFieldInDefaultBaseUnits*cfE*cfT/(cfC*cfL*cfL);
}

double UnitHandler::convertMagneticFieldNaturalToDerived(
	double field,
	MagneticFieldUnit unit
){
	double cfE = getConversionFactor<Quantity::Energy>(
		)/getScale<Quantity::Energy>();
	double cfT = getConversionFactor<Quantity::Time>(
		)/getScale<Quantity::Time>();
	double cfC = getConversionFactor<Quantity::Charge>(
		)/getScale<Quantity::Charge>();
	double cfL = getConversionFactor<Quantity::Length>(
		)/getScale<Quantity::Length>();
	double magneticFieldInDefaultBaseUnits = field*cfC*cfL*cfL/(cfE*cfT);
	return magneticFieldInDefaultBaseUnits*getMagneticFieldConversionFactor(unit);
}

double UnitHandler::convertVoltageDerivedToBase(
	double voltage,
	VoltageUnit unit
){
	double voltageInDefaultBaseUnits = voltage/getVoltageConversionFactor(unit);
	double cfE = getConversionFactor<Quantity::Energy>();
	double cfC = getConversionFactor<Quantity::Charge>();
	return voltageInDefaultBaseUnits*cfE/cfC;
}

double UnitHandler::convertVoltageBaseToDerived(
	double voltage,
	VoltageUnit unit
){
	double cfE = getConversionFactor<Quantity::Energy>();
	double cfC = getConversionFactor<Quantity::Charge>();
	double voltageInDefaultBaseUnits = voltage*cfC/cfE;
	return voltageInDefaultBaseUnits*getVoltageConversionFactor(unit);
}

double UnitHandler::convertVoltageDerivedToNatural(
	double voltage,
	VoltageUnit unit
){
	double voltageInDefaultBaseUnits = voltage/getVoltageConversionFactor(unit);
	double cfE = getConversionFactor<Quantity::Energy>(
		)/getScale<Quantity::Energy>();
	double cfC = getConversionFactor<Quantity::Charge>(
		)/getScale<Quantity::Charge>();
	return voltageInDefaultBaseUnits*cfE/cfC;
}

double UnitHandler::convertVoltageNaturalToDerived(
	double voltage,
	VoltageUnit unit
){
	double cfE = getConversionFactor<Quantity::Energy>(
		)/getScale<Quantity::Energy>();
	double cfC = getConversionFactor<Quantity::Charge>(
		)/getScale<Quantity::Charge>();
	double voltageInDefaultBaseUnits = voltage*cfC/cfE;
	return voltageInDefaultBaseUnits*getVoltageConversionFactor(unit);
}*/

string UnitHandler::getMassUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Energy>() << getUnitString<Quantity::Time>() << "^2" << "/" << getUnitString<Quantity::Length>() << "^2";

	return ss.str();
}

string UnitHandler::getMagneticFieldUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Energy>() << getUnitString<Quantity::Time>() << "/" << getUnitString<Quantity::Charge>() << getUnitString<Quantity::Length>() << "^2";

	return ss.str();
}

string UnitHandler::getVoltageUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Energy>() << "/" << getUnitString<Quantity::Charge>();

	return ss.str();
}

string UnitHandler::getHBARUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Energy>() << getUnitString<Quantity::Time>();

	return ss.str();
}

string UnitHandler::getK_BUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Energy>() << "/" << getUnitString<Quantity::Temperature>();

	return ss.str();
}

string UnitHandler::getEUnitString(){
	return getUnitString<Quantity::Charge>();
}

string UnitHandler::getCUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Length>() << "^2" << "/" << getUnitString<Quantity::Time>() << "^2";

	return ss.str();
}

string UnitHandler::getN_AUnitString(){
	return getUnitString<Quantity::Count>();
}

string UnitHandler::getM_eUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Energy>() << getUnitString<Quantity::Time>() << "^2" << "/" << getUnitString<Quantity::Length>() << "^2";

	return ss.str();
}

string UnitHandler::getM_pUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Energy>() << getUnitString<Quantity::Time>() << "^2" << "/" << getUnitString<Quantity::Length>() << "^2";

	return ss.str();
}

string UnitHandler::getMu_BUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Charge>() << getUnitString<Quantity::Length>() << "^2" << "/" << getUnitString<Quantity::Time>();

	return ss.str();
}

string UnitHandler::getMu_nUnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Charge>() << getUnitString<Quantity::Length>() << "^2" << "/" << getUnitString<Quantity::Time>();

	return ss.str();
}

string UnitHandler::getMu_0UnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Energy>() << getUnitString<Quantity::Time>() << "^2" << "/" << getUnitString<Quantity::Charge>() << "^2" << getUnitString<Quantity::Length>();

	return ss.str();
}

string UnitHandler::getEpsilon_0UnitString(){
	stringstream ss;
	ss << getUnitString<Quantity::Charge>() << "^2" << "/" << getUnitString<Quantity::Energy>() << getUnitString<Quantity::Length>();

	return ss.str();
}

string UnitHandler::getA_0UnitString(){
	return getUnitString<Quantity::Length>();
}

void UnitHandler::updateConstants(){
	constantsBaseUnits.clear();
	for(auto constant : constantsDefaultUnits){
		const string &name = constant.first;
		double value = constant.second.first;
		const vector<pair<string, int>> &units
			= constant.second.second;
		for(unsigned int n = 0; n < units.size(); n++){
			const string &unit = units[n].first;
			int exponent = units[n].second;
			for(int c = 0; c < exponent; c++){
				if(unit.compare("K") == 0){
					value *= getConversionFactor<
						Quantity::Temperature
					>();
				}
				else if(unit.compare("s") == 0){
					value *= getConversionFactor<
						Quantity::Time
					>();
				}
				else if(unit.compare("m") == 0){
					value *= getConversionFactor<
						Quantity::Length
					>();
				}
				else if(unit.compare("eV") == 0){
					value *= getConversionFactor<
						Quantity::Energy
					>();
				}
				else if(unit.compare("C") == 0){
					value *= getConversionFactor<
						Quantity::Charge
					>();
				}
				else if(unit.compare("pcs") == 0){
					value *= getConversionFactor<
						Quantity::Count
					>();
				}
				else{
					TBTKExit(
						"UnitHandler::updateConstants()",
						"Unknown default unit.",
						"This should never happen,"
						<< " contact the developer."
					);
				}
			}
			for(int c = 0; c < -exponent; c++){
				if(unit.compare("K") == 0){
					value /= getConversionFactor<
						Quantity::Temperature
					>();
				}
				else if(unit.compare("s") == 0){
					value /= getConversionFactor<
						Quantity::Time
					>();
				}
				else if(unit.compare("m") == 0){
					value /= getConversionFactor<
						Quantity::Length
					>();
				}
				else if(unit.compare("eV") == 0){
					value /= getConversionFactor<
						Quantity::Energy
					>();
				}
				else if(unit.compare("C") == 0){
					value /= getConversionFactor<
						Quantity::Charge
					>();
				}
				else if(unit.compare("pcs") == 0){
					value /= getConversionFactor<
						Quantity::Count
					>();
				}
				else{
					TBTKExit(
						"UnitHandler::updateConstants()",
						"Unknown default unit.",
						"This should never happen,"
						<< " contact the developer."
					);
				}
			}
		}
		constantsBaseUnits[name] = value;
	}
}

/*double UnitHandler::getMassConversionFactor(MassUnit unit){
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
			TBTKExit(
				"UnitHandler::getMassConversionFactor()",
				"Unknown unit - " << static_cast<int>(unit) << ".",
				""
			);
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
			TBTKExit(
				"UnitHandler::getMagneticFieldConversionFactor()",
				"Unknown unit - " << static_cast<int>(unit) << ".",
				""
			);
	}
}

double UnitHandler::getVoltageConversionFactor(VoltageUnit unit){
	switch(unit){
		case VoltageUnit::GV:
			return V_per_baseVoltage*1e-9;
		case VoltageUnit::MV:
			return V_per_baseVoltage*1e-6;
		case VoltageUnit::kV:
			return V_per_baseVoltage*1e-3;
		case VoltageUnit::V:
			return V_per_baseVoltage;
		case VoltageUnit::mV:
			return V_per_baseVoltage*1e3;
		case VoltageUnit::uV:
			return V_per_baseVoltage*1e6;
		case VoltageUnit::nV:
			return V_per_baseVoltage*1e9;
		default:	//Should never happen, hard error generated for quick bug detection
			TBTKExit(
				"UnitHandler::getVoltageConversionFactor()",
				"Unknown unit - " << static_cast<int>(unit) << ".",
				""
			);
	}
}*/

UnitHandler::StaticConstructor::StaticConstructor(){
	constantsDefaultUnits = {
		{"e",		{1.602176634e-19,			{{"C", 1}}}},
		{"c",		{2.99792458e8,				{{"m", 1}, {"s", -1}}}},
		{"N_A",		{6.02214076e23,				{{"pcs", 1}}}},
		{"a_0",		{5.29177210903*1e-11, 			{{"m", 1}}}}
	};
	double e = constantsDefaultUnits.at("e").first;
	//Source "The International System of Units (SI) 9th Edition. Bureau
	//International des Poids et Mesures. 2019."
	constantsDefaultUnits["h"] =		{6.62607015e-34/e,		{{"eV", 1}, {"s", 1}}};
	constantsDefaultUnits["k_B"] =		{1.380649e-23/e,		{{"eV", 1}, {"K", -1}}};

	double c = constantsDefaultUnits.at("c").first;
	//Source "The NIST reference on Constants, Units, and Uncertainty."
	//https://physics.nist.gov/cuu/Constants/index.html
	constantsDefaultUnits["m_e"] =		{9.1093837015e-31/e,		{{"eV", 1}, {"s", 2}, {"m", -2}}};
	constantsDefaultUnits["m_p"] =		{1.67262192369e-27/e,		{{"eV", 1}, {"s", 2}, {"m", -2}}};
	constantsDefaultUnits["mu_0"] =		{1.25663706212e-6/e,		{{"eV", 1}, {"s", 2}, {"C", -2}, {"m", -1}}};
	constantsDefaultUnits["epsilon_0"] =	{8.8541878128e-12*e,		{{"C", 2}, {"eV", -1}, {"m", -1}}};

	double h = constantsDefaultUnits.at("h").first;
	constantsDefaultUnits["hbar"] =		{h/(2*M_PI),			{{"eV", 1}, {"s", 1}}};
	double hbar = constantsDefaultUnits.at("hbar").first;
	double m_e = constantsDefaultUnits.at("m_e").first;
	double m_p = constantsDefaultUnits.at("m_p").first;
	constantsDefaultUnits["mu_B"] =		{e*hbar/(2*m_e),		{{"C", 1}, {"m", 2}, {"s", -1}}};
	constantsDefaultUnits["mu_N"] =		{e*hbar/(2*m_p),		{{"C", 1}, {"m", 2}, {"s", -1}}};

	kg_per_baseMass			= e;
	baseMass_per_kg			= 1./kg_per_baseMass;
	u_per_baseMass			= (c*c)/9.31494095e8;
	baseMass_per_u			= 1./u_per_baseMass;
	T_per_baseMagneticField		= e;
	baseMagneticField_per_T		= 1./T_per_baseMagneticField;
	V_per_baseVoltage		= e;
	baseVoltage_per_V		= 1./V_per_baseVoltage;

	updateConstants();
}

};

#ifdef M_E_temp
	#define M_E M_E_temp
	#undef M_E_temp
#endif
