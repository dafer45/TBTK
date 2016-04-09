/** @file UnitHandler.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/UnitHandler.h"
#include <sstream>
#include <iostream>

using namespace std;

namespace TBTK{

double UnitHandler::hbar = HBAR;
double UnitHandler::k_b = K_B;

UnitHandler::TemperatureUnit 	UnitHandler::temperatureUnit	= UnitHandler::TemperatureUnit::K;
UnitHandler::TimeUnit 		UnitHandler::timeUnit		= UnitHandler::TimeUnit::s;
UnitHandler::LengthUnit		UnitHandler::lengthUnit		= UnitHandler::LengthUnit::m;
UnitHandler::EnergyUnit		UnitHandler::energyUnit		= UnitHandler::EnergyUnit::eV;

double UnitHandler::temperatureScale = 1.;
double UnitHandler::timeScale = 1.;
double UnitHandler::lengthScale = 1.;
double UnitHandler::energyScale = 1.;

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
}

void UnitHandler::setLengthUnit(LengthUnit unit){
	double oldConversionFactor = getLengthConversionFactor();
	lengthUnit = unit;
	double newConversionFactor = getLengthConversionFactor();
	lengthScale *= newConversionFactor/oldConversionFactor;
}

void UnitHandler::setEnergyUnit(EnergyUnit unit){
	double oldConversionFactor = getEnergyConversionFactor();
	energyUnit = unit;
	double newConversionFactor = getEnergyConversionFactor();
	energyScale *= newConversionFactor/oldConversionFactor;

	updateHbar();
	updateK_b();
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

string UnitHandler::getHBARUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << getTimeUnitString();;

	return ss.str();
}

string UnitHandler::getK_BUnitString(){
	stringstream ss;
	ss << getEnergyUnitString() << "/" << getTemperatureUnitString();

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

double UnitHandler::getTemperatureConversionFactor(){
	switch(temperatureUnit){
		case TemperatureUnit::K:	//Reference scale
			return 1.;
		default:	//Should never happen, hard error generated for quick bug detection
			cout << "Error in UnitHandler::getTemperatureConversionUnit(): Unknown unit - " << temperatureUnit;
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
			cout << "Error in UnitHandler::getTimeConversionFactor(): Unknown unit - " << temperatureUnit;
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
			cout << "Error in UnitHandler::getLengthConversionFactor(): Unknown unit - " << temperatureUnit;
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
			cout << "Error in UnitHandler::getEnergyConversionFactor(): Unknown unit - " << temperatureUnit;
			exit(1);
			return 0.;	//Never happens
	}
}

};
