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

double UnitHandler::J_per_eV;
double UnitHandler::eV_per_J;
double UnitHandler::kg_per_baseMass;
double UnitHandler::baseMass_per_kg;
double UnitHandler::u_per_baseMass;
double UnitHandler::baseMass_per_u;
double UnitHandler::T_per_baseMagneticField;
double UnitHandler::baseMagneticField_per_T;
double UnitHandler::V_per_baseVoltage;
double UnitHandler::baseVoltage_per_V;

Quantity::Temperature::Unit 	UnitHandler::temperatureUnit	= Quantity::Temperature::Unit::K;
Quantity::Time::Unit 		UnitHandler::timeUnit		= Quantity::Time::Unit::s;
Quantity::Length::Unit		UnitHandler::lengthUnit		= Quantity::Length::Unit::m;
Quantity::Energy::Unit		UnitHandler::energyUnit		= Quantity::Energy::Unit::eV;
Quantity::Charge::Unit		UnitHandler::chargeUnit		= Quantity::Charge::Unit::C;
Quantity::Count::Unit		UnitHandler::countUnit		= Quantity::Count::Unit::pcs;

double UnitHandler::temperatureScale	= 1.;
double UnitHandler::timeScale		= 1.;
double UnitHandler::lengthScale		= 1.;
double UnitHandler::energyScale		= 1.;
double UnitHandler::chargeScale		= 1.;
double UnitHandler::countScale		= 1.;

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
				value /= temperatureScale;
			}
			else if(unit.compare("s") == 0){
				value /= timeScale;
			}
			else if(unit.compare("m") == 0){
				value /= lengthScale;
			}
			else if(unit.compare("eV") == 0){
				value /= energyScale;
			}
			else if(unit.compare("C") == 0){
				value /= chargeScale;
			}
			else if(unit.compare("pcs") == 0){
				value /= countScale;
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
				value *= temperatureScale;
			}
			else if(unit.compare("s") == 0){
				value *= timeScale;
			}
			else if(unit.compare("m") == 0){
				value *= lengthScale;
			}
			else if(unit.compare("eV") == 0){
				value *= energyScale;
			}
			else if(unit.compare("C") == 0){
				value *= chargeScale;
			}
			else if(unit.compare("pcs") == 0){
				value *= countScale;
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

void UnitHandler::setTemperatureUnit(Quantity::Temperature::Unit unit){
	double oldConversionFactor = getTemperatureConversionFactor();
	temperatureUnit = unit;
	double newConversionFactor = getTemperatureConversionFactor();
	temperatureScale *= newConversionFactor/oldConversionFactor;

	updateConstants();
}

void UnitHandler::setTimeUnit(Quantity::Time::Unit unit){
	double oldConversionFactor = getTimeConversionFactor();
	timeUnit = unit;
	double newConversionFactor = getTimeConversionFactor();
	timeScale *= newConversionFactor/oldConversionFactor;

	updateConstants();
}

void UnitHandler::setLengthUnit(Quantity::Length::Unit unit){
	double oldConversionFactor = getLengthConversionFactor();
	lengthUnit = unit;
	double newConversionFactor = getLengthConversionFactor();
	lengthScale *= newConversionFactor/oldConversionFactor;

	updateConstants();
}

void UnitHandler::setEnergyUnit(Quantity::Energy::Unit unit){
	double oldConversionFactor = getEnergyConversionFactor();
	energyUnit = unit;
	double newConversionFactor = getEnergyConversionFactor();
	energyScale *= newConversionFactor/oldConversionFactor;

	updateConstants();
}

void UnitHandler::setChargeUnit(Quantity::Charge::Unit unit){
	double oldConversionFactor = getChargeConversionFactor();
	chargeUnit = unit;
	double newConversionFactor = getChargeConversionFactor();
	chargeScale *= newConversionFactor/oldConversionFactor;

	updateConstants();
}

void UnitHandler::setCountUnit(Quantity::Count::Unit unit){
	double oldConversionFactor = getCountConversionFactor();
	countUnit = unit;
	double newConversionFactor = getCountConversionFactor();
	countScale *= newConversionFactor/oldConversionFactor;

	updateConstants();
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

void UnitHandler::setCountScale(double scale){
	countScale = scale;
}

void UnitHandler::setTemperatureScale(string scale){
	stringstream ss(scale);
	vector<string> components;
	string word;
	while(getline(ss, word, ' '))
		components.push_back(word);

	TBTKAssert(
		components.size() == 2,
		"UnitHandler::setTemperatureScale()",
		"Invalid temperature scale string '" << scale << "'.",
		"The string has to be on the format '[scale] [unit]', e.g."
		<< " '1 K'."
	);

	double s;
	try{
		s = stod(components[0]);
	}
	catch(const std::exception &e){
		TBTKExit(
			"UnitHandler::setTemperatureScale()",
			"Unable to parse '" << components[0] << "' as a"
			" double.",
			"The string has to be on the format '[scale] [unit]', e.g."
			<< " '1 K'."
		);
	}

	Quantity::Temperature::Unit unit = getTemperatureUnit(components[1]);

	setTemperatureScale(s, unit);
}

void UnitHandler::setTimeScale(string scale){
	stringstream ss(scale);
	vector<string> components;
	string word;
	while(getline(ss, word, ' '))
		components.push_back(word);

	TBTKAssert(
		components.size() == 2,
		"UnitHandler::setTimeScale()",
		"Invalid time scale string '" << scale << "'.",
		"The string has to be on the format '[scale] [unit]', e.g."
		<< " '1 s'."
	);

	double s;
	try{
		s = stod(components[0]);
	}
	catch(const std::exception &e){
		TBTKExit(
			"UnitHandler::setTimeScale()",
			"Unable to parse '" << components[0] << "' as a"
			" double.",
			"The string has to be on the format '[scale] [unit]', e.g."
			<< " '1 s'."
		);
	}

	Quantity::Time::Unit unit = getTimeUnit(components[1]);

	setTimeScale(s, unit);
}

void UnitHandler::setLengthScale(string scale){
	stringstream ss(scale);
	vector<string> components;
	string word;
	while(getline(ss, word, ' '))
		components.push_back(word);

	TBTKAssert(
		components.size() == 2,
		"UnitHandler::setLengthScale()",
		"Invalid length scale string '" << scale << "'.",
		"The string has to be on the format '[scale] [unit]', e.g."
		<< " '1 m'."
	);

	double s;
	try{
		s = stod(components[0]);
	}
	catch(const std::exception &e){
		TBTKExit(
			"UnitHandler::setLengthScale()",
			"Unable to parse '" << components[0] << "' as a"
			" double.",
			"The string has to be on the format '[scale] [unit]', e.g."
			<< " '1 m'."
		);
	}

	Quantity::Length::Unit unit = getLengthUnit(components[1]);

	setLengthScale(s, unit);
}

void UnitHandler::setEnergyScale(string scale){
	stringstream ss(scale);
	vector<string> components;
	string word;
	while(getline(ss, word, ' '))
		components.push_back(word);

	TBTKAssert(
		components.size() == 2,
		"UnitHandler::setEnergyScale()",
		"Invalid energy scale string '" << scale << "'.",
		"The string has to be on the format '[scale] [unit]', e.g."
		<< " '1 eV'."
	);

	double s;
	try{
		s = stod(components[0]);
	}
	catch(const std::exception &e){
		TBTKExit(
			"UnitHandler::setEnergyScale()",
			"Unable to parse '" << components[0] << "' as a"
			" double.",
			"The string has to be on the format '[scale] [unit]', e.g."
			<< " '1 eV'."
		);
	}

	Quantity::Energy::Unit unit = getEnergyUnit(components[1]);

	setEnergyScale(s, unit);
}

void UnitHandler::setChargeScale(string scale){
	stringstream ss(scale);
	vector<string> components;
	string word;
	while(getline(ss, word, ' '))
		components.push_back(word);

	TBTKAssert(
		components.size() == 2,
		"UnitHandler::setChargeScale()",
		"Invalid charge scale string '" << scale << "'.",
		"The string has to be on the format '[scale] [unit]', e.g."
		<< " '1 C'."
	);

	double s;
	try{
		s = stod(components[0]);
	}
	catch(const std::exception &e){
		TBTKExit(
			"UnitHandler::setChargeScale()",
			"Unable to parse '" << components[0] << "' as a"
			" double.",
			"The string has to be on the format '[scale] [unit]', e.g."
			<< " '1 C'."
		);
	}

	Quantity::Charge::Unit unit = getChargeUnit(components[1]);

	setChargeScale(s, unit);
}

void UnitHandler::setCountScale(string scale){
	stringstream ss(scale);
	vector<string> components;
	string word;
	while(getline(ss, word, ' '))
		components.push_back(word);

	TBTKAssert(
		components.size() == 2,
		"UnitHandler::setCountScale()",
		"Invalid count scale string '" << scale << "'.",
		"The string has to be on the format '[scale] [unit]', e.g."
		<< " '1 pcs'."
	);

	double s;
	try{
		s = stod(components[0]);
	}
	catch(const std::exception &e){
		TBTKExit(
			"UnitHandler::setCountScale()",
			"Unable to parse '" << components[0] << "' as a"
			" double.",
			"The string has to be on the format '[scale] [unit]', e.g."
			<< " '1 pcs'."
		);
	}

	Quantity::Count::Unit unit = getCountUnit(components[1]);

	setCountScale(s, unit);
}

double UnitHandler::convertMassDerivedToBase(double mass, MassUnit unit){
	double massInDefaultBaseUnits = mass/getMassConversionFactor(unit);
	double cfE = getEnergyConversionFactor();
	double cfT = getTimeConversionFactor();
	double cfL = getLengthConversionFactor();
	return massInDefaultBaseUnits*cfE*cfT*cfT/(cfL*cfL);
}

double UnitHandler::convertMassBaseToDerived(double mass, MassUnit unit){
	double cfE = getEnergyConversionFactor();
	double cfT = getTimeConversionFactor();
	double cfL = getLengthConversionFactor();
	double massInDefaultBaseUnits = mass*cfL*cfL/(cfE*cfT*cfT);
	return massInDefaultBaseUnits*getMassConversionFactor(unit);
}

double UnitHandler::convertMassDerivedToNatural(double mass, MassUnit unit){
	double massInDefaultBaseUnits = mass/getMassConversionFactor(unit);
	double cfE = getEnergyConversionFactor()/energyScale;
	double cfT = getTimeConversionFactor()/timeScale;
	double cfL = getLengthConversionFactor()/lengthScale;
	return massInDefaultBaseUnits*cfE*cfT*cfT/(cfL*cfL);
}

double UnitHandler::convertMassNaturalToDerived(double mass, MassUnit unit){
	double cfE = getEnergyConversionFactor()/energyScale;
	double cfT = getTimeConversionFactor()/timeScale;
	double cfL = getLengthConversionFactor()/lengthScale;
	double massInDefaultBaseUnits = mass*cfL*cfL/(cfE*cfT*cfT);
	return massInDefaultBaseUnits*getMassConversionFactor(unit);
}

double UnitHandler::convertMagneticFieldDerivedToBase(
	double field,
	MagneticFieldUnit unit
){
	double magneticFieldInDefaultBaseUnits = field/getMagneticFieldConversionFactor(unit);
	double cfE = getEnergyConversionFactor();
	double cfT = getTimeConversionFactor();
	double cfC = getChargeConversionFactor();
	double cfL = getLengthConversionFactor();
	return magneticFieldInDefaultBaseUnits*cfE*cfT/(cfC*cfL*cfL);
}

double UnitHandler::convertMagneticFieldBaseToDerived(
	double field,
	MagneticFieldUnit unit
){
	double cfE = getEnergyConversionFactor();
	double cfT = getTimeConversionFactor();
	double cfC = getChargeConversionFactor();
	double cfL = getLengthConversionFactor();
	double magneticFieldInDefaultBaseUnits = field*cfC*cfL*cfL/(cfE*cfT);
	return magneticFieldInDefaultBaseUnits*getMagneticFieldConversionFactor(unit);
}

double UnitHandler::convertMagneticFieldDerivedToNatural(
	double field,
	MagneticFieldUnit unit
){
	double magneticFieldInDefaultBaseUnits = field/getMagneticFieldConversionFactor(unit);
	double cfE = getEnergyConversionFactor()/energyScale;
	double cfT = getTimeConversionFactor()/timeScale;
	double cfC = getChargeConversionFactor()/chargeScale;
	double cfL = getLengthConversionFactor()/lengthScale;
	return magneticFieldInDefaultBaseUnits*cfE*cfT/(cfC*cfL*cfL);
}

double UnitHandler::convertMagneticFieldNaturalToDerived(
	double field,
	MagneticFieldUnit unit
){
	double cfE = getEnergyConversionFactor()/energyScale;
	double cfT = getTimeConversionFactor()/timeScale;
	double cfC = getChargeConversionFactor()/chargeScale;
	double cfL = getLengthConversionFactor()/lengthScale;
	double magneticFieldInDefaultBaseUnits = field*cfC*cfL*cfL/(cfE*cfT);
	return magneticFieldInDefaultBaseUnits*getMagneticFieldConversionFactor(unit);
}

double UnitHandler::convertVoltageDerivedToBase(
	double voltage,
	VoltageUnit unit
){
	double voltageInDefaultBaseUnits = voltage/getVoltageConversionFactor(unit);
	double cfE = getEnergyConversionFactor();
	double cfC = getChargeConversionFactor();
	return voltageInDefaultBaseUnits*cfE/cfC;
}

double UnitHandler::convertVoltageBaseToDerived(
	double voltage,
	VoltageUnit unit
){
	double cfE = getEnergyConversionFactor();
	double cfC = getChargeConversionFactor();
	double voltageInDefaultBaseUnits = voltage*cfC/cfE;
	return voltageInDefaultBaseUnits*getVoltageConversionFactor(unit);
}

double UnitHandler::convertVoltageDerivedToNatural(
	double voltage,
	VoltageUnit unit
){
	double voltageInDefaultBaseUnits = voltage/getVoltageConversionFactor(unit);
	double cfE = getEnergyConversionFactor()/energyScale;
	double cfC = getChargeConversionFactor()/chargeScale;
	return voltageInDefaultBaseUnits*cfE/cfC;
}

double UnitHandler::convertVoltageNaturalToDerived(
	double voltage,
	VoltageUnit unit
){
	double cfE = getEnergyConversionFactor()/energyScale;
	double cfC = getChargeConversionFactor()/chargeScale;
	double voltageInDefaultBaseUnits = voltage*cfC/cfE;
	return voltageInDefaultBaseUnits*getVoltageConversionFactor(unit);
}

string UnitHandler::getTemperatureUnitString(){
	return Quantity::Temperature::getUnitString(temperatureUnit);
}

string UnitHandler::getTimeUnitString(){
	return Quantity::Time::getUnitString(timeUnit);
}

string UnitHandler::getLengthUnitString(){
	return Quantity::Length::getUnitString(lengthUnit);
}

string UnitHandler::getEnergyUnitString(){
	return Quantity::Energy::getUnitString(energyUnit);
}

string UnitHandler::getChargeUnitString(){
	return Quantity::Charge::getUnitString(chargeUnit);
}

string UnitHandler::getCountUnitString(){
	return Quantity::Count::getUnitString(countUnit);
}

string UnitHandler::getMassUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << getTimeUnitString() << "^2" << "/" << getLengthUnitString() << "^2";

	return ss.str();
}

string UnitHandler::getMagneticFieldUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << getTimeUnitString() << "/" << getChargeUnitString() << getLengthUnitString() << "^2";

	return ss.str();
}

string UnitHandler::getVoltageUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << "/" << getChargeUnitString();

	return ss.str();
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
	ss << getLengthUnitString() << "^2" << "/" << getTimeUnitString() << "^2";

	return ss.str();
}

string UnitHandler::getN_AUnitString(){
	return getCountUnitString();
}

string UnitHandler::getM_eUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << getTimeUnitString() << "^2" << "/" << getLengthUnitString() << "^2";

	return ss.str();
}

string UnitHandler::getM_pUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << getTimeUnitString() << "^2" << "/" << getLengthUnitString() << "^2";

	return ss.str();
}

string UnitHandler::getMu_BUnitString(){
	stringstream ss;
	ss << getChargeUnitString() << getLengthUnitString() << "^2" << "/" << getTimeUnitString();

	return ss.str();
}

string UnitHandler::getMu_nUnitString(){
	stringstream ss;
	ss << getChargeUnitString() << getLengthUnitString() << "^2" << "/" << getTimeUnitString();

	return ss.str();
}

string UnitHandler::getMu_0UnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << getTimeUnitString() << "^2" << "/" << getChargeUnitString() << "^2" << getLengthUnitString();

	return ss.str();
}

string UnitHandler::getEpsilon_0UnitString(){
	stringstream ss;
	ss << getChargeUnitString() << "^2" << "/" << getEnergyUnitString() << getLengthUnitString();

	return ss.str();
}

string UnitHandler::getA_0UnitString(){
	return getLengthUnitString();
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
					value *= getTemperatureConversionFactor();
				}
				else if(unit.compare("s") == 0){
					value *= getTimeConversionFactor();
				}
				else if(unit.compare("m") == 0){
					value *= getLengthConversionFactor();
				}
				else if(unit.compare("eV") == 0){
					value *= getEnergyConversionFactor();
				}
				else if(unit.compare("C") == 0){
					value *= getChargeConversionFactor();
				}
				else if(unit.compare("pcs") == 0){
					value *= getCountConversionFactor();
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
					value /= getTemperatureConversionFactor();
				}
				else if(unit.compare("s") == 0){
					value /= getTimeConversionFactor();
				}
				else if(unit.compare("m") == 0){
					value /= getLengthConversionFactor();
				}
				else if(unit.compare("eV") == 0){
					value /= getEnergyConversionFactor();
				}
				else if(unit.compare("C") == 0){
					value /= getChargeConversionFactor();
				}
				else if(unit.compare("pcs") == 0){
					value /= getCountConversionFactor();
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

double UnitHandler::getTemperatureConversionFactor(){
	return getTemperatureConversionFactor(temperatureUnit);
}

double UnitHandler::getTemperatureConversionFactor(
	Quantity::Temperature::Unit temperatureUnit
){
	switch(temperatureUnit){
		case Quantity::Temperature::Unit::kK:	//1e-3 kK per K
			return 1e-3;
		case Quantity::Temperature::Unit::K:	//Reference scale
			return 1.;
		case Quantity::Temperature::Unit::mK:
			return 1e3;
		case Quantity::Temperature::Unit::uK:
			return 1e6;
		case Quantity::Temperature::Unit::nK:
			return 1e9;
		default:	//Should never happen, hard error generated for quick bug detection
			TBTKExit(
				"UnitHandler::getTemperatureConversionUnit()",
				"Unknown unit - " << static_cast<int>(temperatureUnit) << ".",
				""
			);
	}
}

double UnitHandler::getTimeConversionFactor(){
	return getTimeConversionFactor(timeUnit);
}

double UnitHandler::getTimeConversionFactor(Quantity::Time::Unit timeUnit){
	switch(timeUnit){
		case Quantity::Time::Unit::s:	//Reference scale
			return 1.;
		case Quantity::Time::Unit::ms:
			return 1e3;	//1e3 ms per second
		case Quantity::Time::Unit::us:
			return 1e6;
		case Quantity::Time::Unit::ns:
			return 1e9;
		case Quantity::Time::Unit::ps:
			return 1e12;
		case Quantity::Time::Unit::fs:
			return 1e15;
		case Quantity::Time::Unit::as:
			return 1e18;
		default:	//Should never happen, hard error generated for quick bug detection
			TBTKExit(
				"UnitHandler::getTimeConversionFactor()",
				"Unknown unit - " << static_cast<int>(timeUnit) << ".",
				""
			);
	}
}

double UnitHandler::getLengthConversionFactor(){
	return getLengthConversionFactor(lengthUnit);
}

double UnitHandler::getLengthConversionFactor(
	Quantity::Length::Unit lengthUnit
){
	switch(lengthUnit){
		case Quantity::Length::Unit::m:	//Reference scale
			return 1.;
		case Quantity::Length::Unit::mm:	//1e3 mm per m
			return 1e3;
		case Quantity::Length::Unit::um:
			return 1e6;
		case Quantity::Length::Unit::nm:
			return 1e9;
		case Quantity::Length::Unit::pm:
			return 1e12;
		case Quantity::Length::Unit::fm:
			return 1e15;
		case Quantity::Length::Unit::am:
			return 1e18;
		case Quantity::Length::Unit::Ao:
			return 1e10;
		default:	//Should never happen, hard error generated for quick bug detection
			TBTKExit(
				"UnitHandler::getLengthConversionFactor()",
				"Unknown unit - " << static_cast<int>(lengthUnit) << ".",
				""
			);
	}
}

double UnitHandler::getEnergyConversionFactor(){
	return getEnergyConversionFactor(energyUnit);
}

double UnitHandler::getEnergyConversionFactor(
	Quantity::Energy::Unit energyUnit
){
	switch(energyUnit){
		case Quantity::Energy::Unit::GeV:	//1e-9 GeV per eV
			return 1e-9;
		case Quantity::Energy::Unit::MeV:
			return 1e-6;
		case Quantity::Energy::Unit::keV:
			return 1e-3;
		case Quantity::Energy::Unit::eV:	//Reference scale
			return 1.;
		case Quantity::Energy::Unit::meV:
			return 1e3;
		case Quantity::Energy::Unit::ueV:
			return 1e6;
		case Quantity::Energy::Unit::J:
			return J_per_eV;
		default:	//Should never happen, hard error generated for quick bug detection
			TBTKExit(
				"UnitHandler::getEnergyConversionFactor()",
				"Unknown unit - " << static_cast<int>(energyUnit) << ".",
				""
			);
	}
}

double UnitHandler::getChargeConversionFactor(){
	return getChargeConversionFactor(chargeUnit);
}

double UnitHandler::getChargeConversionFactor(
	Quantity::Charge::Unit chargeUnit
){
	double E = constantsDefaultUnits.at("e").first;
	switch(chargeUnit){
		case Quantity::Charge::Unit::kC:	//1e-3 kC per C
			return 1e-3;
		case Quantity::Charge::Unit::C:	//Reference scale
			return 1.;
		case Quantity::Charge::Unit::mC:
			return 1e3;
		case Quantity::Charge::Unit::uC:
			return 1e6;
		case Quantity::Charge::Unit::nC:
			return 1e9;
		case Quantity::Charge::Unit::pC:
			return 1e12;
		case Quantity::Charge::Unit::fC:
			return 1e15;
		case Quantity::Charge::Unit::aC:
			return 1e18;
		case Quantity::Charge::Unit::Te:
			return 1e-12/E;
		case Quantity::Charge::Unit::Ge:
			return 1e-9/E;
		case Quantity::Charge::Unit::Me:
			return 13-6/E;
		case Quantity::Charge::Unit::ke:
			return 1e-3/E;
		case Quantity::Charge::Unit::e:
			return 1./E;
		default:	//Should never happen, hard error generated for quick bug detection
			TBTKExit(
				"UnitHandler::getChargeConversionFactor()",
				"Unknown unit - " << static_cast<int>(chargeUnit) << ".",
				""
			);
	}
}

double UnitHandler::getCountConversionFactor(){
	return getCountConversionFactor(countUnit);
}

double UnitHandler::getCountConversionFactor(Quantity::Count::Unit countUnit){
	double N_A = constantsDefaultUnits.at("N_A").first;
	switch(countUnit){
		case Quantity::Count::Unit::pcs:
			return 1.;	//Reference scale
		case Quantity::Count::Unit::mol:	//1/N_A mol per pcs
			return 1./N_A;
		default:	//Should never happen, hard error generated for quick bug detection
			TBTKExit(
				"UnitHandler::getCountConversionFactor()",
				"Unknown unit - " << static_cast<int>(countUnit) << ".",
				""
			);
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
}

Quantity::Temperature::Unit UnitHandler::getTemperatureUnit(string unit){
	if(unit.compare("kK") == 0){
		return Quantity::Temperature::Unit::kK;
	}
	else if(unit.compare("K") == 0){
		return Quantity::Temperature::Unit::K;
	}
	else if(unit.compare("mK") == 0){
		return Quantity::Temperature::Unit::mK;
	}
	else if(unit.compare("uK") == 0){
		return Quantity::Temperature::Unit::uK;
	}
	else if(unit.compare("nK") == 0){
		return Quantity::Temperature::Unit::nK;
	}
	else{
		TBTKExit(
			"UnitHandler::getTemperatureUnit()",
			"Invalid temperature unit '" << unit << "'",
			""
		);
	}
}

Quantity::Time::Unit UnitHandler::getTimeUnit(string unit){
	if(unit.compare("s") == 0){
		return Quantity::Time::Unit::s;
	}
	else if(unit.compare("ms") == 0){
		return Quantity::Time::Unit::ms;
	}
	else if(unit.compare("us") == 0){
		return Quantity::Time::Unit::us;
	}
	else if(unit.compare("ns") == 0){
		return Quantity::Time::Unit::ns;
	}
	else if(unit.compare("ps") == 0){
		return Quantity::Time::Unit::ps;
	}
	else if(unit.compare("fs") == 0){
		return Quantity::Time::Unit::fs;
	}
	else if(unit.compare("as") == 0){
		return Quantity::Time::Unit::as;
	}
	else{
		TBTKExit(
			"UnitHandler::getTimeUnit()",
			"Invalid time unit '" << unit << "'",
			""
		);
	}
}

Quantity::Length::Unit UnitHandler::getLengthUnit(string unit){
	if(unit.compare("m") == 0){
		return Quantity::Length::Unit::m;
	}
	else if(unit.compare("mm") == 0){
		return Quantity::Length::Unit::mm;
	}
	else if(unit.compare("um") == 0){
		return Quantity::Length::Unit::um;
	}
	else if(unit.compare("nm") == 0){
		return Quantity::Length::Unit::nm;
	}
	else if(unit.compare("pm") == 0){
		return Quantity::Length::Unit::pm;
	}
	else if(unit.compare("fm") == 0){
		return Quantity::Length::Unit::fm;
	}
	else if(unit.compare("am") == 0){
		return Quantity::Length::Unit::am;
	}
	else if(unit.compare("Ao") == 0){
		return Quantity::Length::Unit::Ao;
	}
	else{
		TBTKExit(
			"UnitHandler::getLengthUnit()",
			"Invalid length unit '" << unit << "'",
			""
		);
	}
}

Quantity::Energy::Unit UnitHandler::getEnergyUnit(string unit){
	if(unit.compare("GeV") == 0){
		return Quantity::Energy::Unit::GeV;
	}
	else if(unit.compare("MeV") == 0){
		return Quantity::Energy::Unit::MeV;
	}
	else if(unit.compare("keV") == 0){
		return Quantity::Energy::Unit::keV;
	}
	else if(unit.compare("eV") == 0){
		return Quantity::Energy::Unit::eV;
	}
	else if(unit.compare("meV") == 0){
		return Quantity::Energy::Unit::meV;
	}
	else if(unit.compare("ueV") == 0){
		return Quantity::Energy::Unit::ueV;
	}
	else if(unit.compare("J") == 0){
		return Quantity::Energy::Unit::J;
	}
	else{
		TBTKExit(
			"UnitHandler::getEnergyUnit()",
			"Invalid energy unit '" << unit << "'",
			""
		);
	}
}

Quantity::Charge::Unit UnitHandler::getChargeUnit(string unit){
	if(unit.compare("kC") == 0){
		return Quantity::Charge::Unit::kC;
	}
	else if(unit.compare("C") == 0){
		return Quantity::Charge::Unit::C;
	}
	else if(unit.compare("mC") == 0){
		return Quantity::Charge::Unit::mC;
	}
	else if(unit.compare("uC") == 0){
		return Quantity::Charge::Unit::uC;
	}
	else if(unit.compare("nC") == 0){
		return Quantity::Charge::Unit::nC;
	}
	else if(unit.compare("pC") == 0){
		return Quantity::Charge::Unit::pC;
	}
	else if(unit.compare("fC") == 0){
		return Quantity::Charge::Unit::fC;
	}
	else if(unit.compare("aC") == 0){
		return Quantity::Charge::Unit::aC;
	}
	else if(unit.compare("Te") == 0){
		return Quantity::Charge::Unit::Te;
	}
	else if(unit.compare("Ge") == 0){
		return Quantity::Charge::Unit::Ge;
	}
	else if(unit.compare("Me") == 0){
		return Quantity::Charge::Unit::Me;
	}
	else if(unit.compare("ke") == 0){
		return Quantity::Charge::Unit::ke;
	}
	else if(unit.compare("e") == 0){
		return Quantity::Charge::Unit::e;
	}
	else{
		TBTKExit(
			"UnitHandler::getChargeUnit()",
			"Invalid charge unit '" << unit << "'",
			""
		);
	}
}

Quantity::Count::Unit UnitHandler::getCountUnit(string unit){
	if(unit.compare("pcs") == 0){
		return Quantity::Count::Unit::pcs;
	}
	else if(unit.compare("mol") == 0){
		return Quantity::Count::Unit::mol;
	}
	else{
		TBTKExit(
			"UnitHandler::getCountUnit()",
			"Invalid count unit '" << unit << "'",
			""
		);
	}
}

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

	J_per_eV			= e;
	eV_per_J			= 1./J_per_eV;
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
