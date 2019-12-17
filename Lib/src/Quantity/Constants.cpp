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

//Source "The International System of Units (SI) 9th Edition. Bureau
//International des Poids et Mesures. 2019."
/*pair<string, Constant> Constants::e;
pair<string, Velocity> Constants::c;
pair<string, Count> Constants::N_A;
pair<string, Length> Constants::a_0;
pair<string, Planck> Constants::h;
pair<string, Boltzmann> Constants::k_B;*/

//Source "The NIST reference on Constants, Units, and Uncertainty."
//https://physics.nist.gov/cuu/Constants/index.html.
/*pair<string, Mass> Constants::m_e;
pair<string, Mass> Constants::m_p;
pair<string, Permeability> Constants::mu_0;
pair<string, Permittivity> Constants::epsilon_0;*/

//Calculated values.
/*pair<string, Planck> Constants::hbar;
pair<string, Magneton> Constants::mu_B;
pair<string, Magneton> Constants::mu_N;*/

void Constants::initialize(){
	//Source "The International System of Units (SI) 9th Edition. Bureau
	//International des Poids et Mesures. 2019."
	constants["e"] = Constant(1.602176634e-19, Charge());
	constants["c"] = Constant(2.99792458e8, Velocity());
	constants["N_A"] = Constant(6.02214076e23, Count());
	constants["a_0"] = Constant(5.29177210903*1e-11, Length());
	constants["h"] = Constant(6.62607015e-34/constants["e"], Planck());
	constants["k_B"] = Constant(1.380649e-23/constants["e"], Boltzmann());
/*	e = {"e", Constant(1.602176634e-19, Charge())};
	c = {"c", 2.99792458e8};
	N_A = {"N_A", 6.02214076e23};
	a_0 = {"a_0", 5.29177210903*1e-11};
	h = {"h", 6.62607015e-34/Constants::e.second};
	k_B = {"k_B", 1.380649e-23/Constants::e.second};*/

	//Source "The NIST reference on Constants, Units, and Uncertainty."
	//https://physics.nist.gov/cuu/Constants/index.html.
	constants["m_e"] = Constant(9.1093837015e-31/constants["e"], Mass());
	constants["m_p"] = Constant(1.67262192369e-27/constants["e"], Mass());
	constants["mu_0"] = Constant(1.25663706212e-6/constants["e"], Permeability());
	constants["epsilon_0"] = Constant(8.8541878128e-12*constants["e"], Permittivity());
/*	m_e = {"m_e", 9.1093837015e-31/e.second};
	m_p = {"m_p", 1.67262192369e-27/e.second};
	mu_0 = {"mu_0", 1.25663706212e-6/e.second};
	epsilon_0 = {"epsilon_0", 8.8541878128e-12*e.second};*/

	//Calculated values.
	constants["hbar"] = Constant(constants["h"]/(2*M_PI), Planck());
	constants["mu_B"] = Constant(constants["e"]*constants["hbar"]/(2*constants["m_e"]), Magneton());
	constants["mu_N"] = Constant(constants["e"]*constants["hbar"]/(2*constants["m_p"]), Magneton());
/*	hbar = {"hbar", h.second/(2*M_PI)};
	mu_B = {"mu_B", e.second*hbar.second/(2*m_e.second)};
	mu_N = {"mu_N", e.second*hbar.second/(2*m_p.second)};*/
}

};	//End of namespace Quantity
};	//End of namespace TBTK
