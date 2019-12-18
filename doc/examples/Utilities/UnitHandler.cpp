#include "HeaderAndFooter.h"
TBTK::DocumentationExamples::HeaderAndFooter headerAndFooter("UnitHandler");

//! [UnitHandler]
#include "TBTK/UnitHandler.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTK.h"

using namespace TBTK;

int main(){
	Initialize();

	//Initialize the scales. The base units are set to rad, C, pcs, eV, m,
	//K, and s. The natural length scale is set to 1 feet.
	UnitHandler::setScales(
		{"1 rad", "1 C", "1 pcs", "1 eV", "0.3048 m", "1 K", "1 s"}
	);

	//Print the speed of light in the base units (m/s).
	Streams::out << "The speed of light is "
		<< UnitHandler::getConstantInBaseUnits("c") << " "
		<< UnitHandler::getUnitString("c") << ".\n";

	//Print the speed of light in the natural units (ft/s).
	Streams::out << "The speed of light is "
		<< UnitHandler::getConstantInNaturalUnits("c")
		<< " ft s^-1.\n";

	//Convert from natural to base units.
	Streams::out << "10 m is equal to "
		<< UnitHandler::convertBaseToNatural<Quantity::Length>(10)
		<< " ft.\n";

	//Convert from base to natural units.
	Streams::out << "10 ft is equal to "
		<< UnitHandler::convertNaturalToBase<Quantity::Length>(10)
		<< UnitHandler::getUnitString<Quantity::Length>() << ".\n";

	//Print the reduced Planck constant in base units.
	Streams::out << "The reduced Planck constant is "
		<< UnitHandler::getConstantInBaseUnits("hbar") << " "
		<< UnitHandler::getUnitString("hbar") << "\n";

	//Convert 1 V to base units.
	Streams::out << "1 V is equal to "
		<< UnitHandler::convertArbitraryToBase<Quantity::Voltage>(
			1,
			Quantity::Voltage::Unit::V
		) << " " << UnitHandler::getUnitString<Quantity::Voltage>()
		<< ".\n";

	//Convert one natural unit of voltage to V.
	Streams::out << "1 natural unit of voltage is equal to "
		<< UnitHandler::convertNaturalToArbitrary<Quantity::Voltage>(
			1,
			Quantity::Voltage::Unit::V
		) << " V\n.";
}
//! [UnitHandler]
