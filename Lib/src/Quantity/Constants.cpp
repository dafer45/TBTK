/* Copyright 2019 Kristofer Björnson
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

/** @file Constants.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Quantity/Constants.h"

using namespace std;

namespace TBTK{
namespace Quantity{

map<string, Constant> Constants::constants;

void Constants::initialize(){
	//Source "The International System of Units (SI) 9th Edition. Bureau
	//International des Poids et Mesures. 2019."
	constants["e"] = Constant(1.602176634e-19, Charge());
	constants["c"] = Constant(2.99792458e8, Velocity());
	constants["N_A"] = Constant(6.02214076e23, Count());
	constants["a_0"] = Constant(5.29177210903*1e-11, Length());
	constants["h"] = Constant(6.62607015e-34/constants["e"], Planck());
	constants["k_B"] = Constant(1.380649e-23/constants["e"], Boltzmann());

	//Source "The NIST reference on Constants, Units, and Uncertainty."
	//https://physics.nist.gov/cuu/Constants/index.html.
	constants["m_e"] = Constant(9.1093837015e-31/constants["e"], Mass());
	constants["m_p"] = Constant(1.67262192369e-27/constants["e"], Mass());
	constants["mu_0"] = Constant(1.25663706212e-6/constants["e"], Permeability());
	constants["epsilon_0"] = Constant(8.8541878128e-12*constants["e"], Permittivity());

	//Calculated values.
	constants["hbar"] = Constant(constants["h"]/(2*M_PI), Planck());
	constants["mu_B"] = Constant(constants["e"]*constants["hbar"]/(2*constants["m_e"]), Magneton());
	constants["mu_N"] = Constant(constants["e"]*constants["hbar"]/(2*constants["m_p"]), Magneton());
}

};	//End of namespace Quantity
};	//End of namespace TBTK
