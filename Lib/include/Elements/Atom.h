/* Copyright 2017 Kristofer Björnson
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

/** @package TBTKcalc
 *  @file Atom.h
 *  @brief Atom.h
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_ATOM
#define COM_DAFER45_TBTK_ATOM

#include "UnitHandler.h"

#include <string>

namespace TBTK{

class Atom{
public:
	/** Constructor. */
	Atom(unsigned int atomicNumber);

	/** Destructor. */
	~Atom();

	/** Get atomic number. */
	unsigned int getAtomicNumber() const;

	/** Get symbol. */
	std::string getSymbol() const;

	/** Get name. */
	std::string getName() const;

	/** Get standard weight. */
	double getStandardWeight() const;

	/** Get Atom by number. */
	static const Atom& getAtomByNumber(unsigned int atomicNumber);

	/** Get Atom by symbol. */
	static const Atom& getAtomBySymbol(const std::string &symbol);

	/** Get Atom by name. */
	static const Atom& getAtomByName(const std::string &name);

	/** Atoms by name. */
	static const Atom &Hydrogen;
	static const Atom &Helium;
	static const Atom &Lithium;
	static const Atom &Beryllium;
	static const Atom &Boron;
	static const Atom &Carbon;
	static const Atom &Nitrogen;
	static const Atom &Oxygen;
	static const Atom &Fluorine;
	static const Atom &Neon;		//10
	static const Atom &Sodium;
	static const Atom &Magnesium;
	static const Atom &Aluminium;
	static const Atom &Silicon;
	static const Atom &Phosphorus;
	static const Atom &Sulfur;
	static const Atom &Chlorine;
	static const Atom &Argon;
	static const Atom &Potassium;
	static const Atom &Calcium;		//20
	static const Atom &Scandium;
	static const Atom &Titanium;
	static const Atom &Vanadium;
	static const Atom &Chromium;
	static const Atom &Manganese;
	static const Atom &Iron;
	static const Atom &Cobalt;
	static const Atom &Nickel;
	static const Atom &Copper;
	static const Atom &Zinc;		//30
	static const Atom &Gallium;
	static const Atom &Germanium;
	static const Atom &Arsenic;
	static const Atom &Selenium;
	static const Atom &Bromine;
	static const Atom &Krypton;
	static const Atom &Rubidium;
	static const Atom &Strontium;
	static const Atom &Yttrium;
	static const Atom &Zirconium;		//40
	static const Atom &Niobium;
	static const Atom &Molybdenum;
	static const Atom &Technetium;
	static const Atom &Ruthenium;
	static const Atom &Rhodium;
	static const Atom &Palladium;
	static const Atom &Silver;
	static const Atom &Cadmium;
	static const Atom &Indium;
	static const Atom &Tin;			//50
	static const Atom &Antimony;
	static const Atom &Tellurium;
	static const Atom &Iodine;
	static const Atom &Xenon;
	static const Atom &Caesium;
	static const Atom &Barium;
	static const Atom &Lanthanum;
	static const Atom &Cerium;
	static const Atom &Praseodymium;
	static const Atom &Neodymium;		//60
	static const Atom &Promethium;
	static const Atom &Samarium;
	static const Atom &Europium;
	static const Atom &Gadolinium;
	static const Atom &Terbium;
	static const Atom &Dysprosium;
	static const Atom &Holmium;
	static const Atom &Erbium;
	static const Atom &Thulium;
	static const Atom &Ytterbium;		//70
	static const Atom &Lutetium;
	static const Atom &Hafnium;
	static const Atom &Tantalum;
	static const Atom &Tungsten;
	static const Atom &Rhenium;
	static const Atom &Osmium;
	static const Atom &Iridium;
	static const Atom &Platinum;
	static const Atom &Gold;
	static const Atom &Mercury;		//80
	static const Atom &Thallium;
	static const Atom &Lead;
	static const Atom &Bismuth;
	static const Atom &Polonium;
	static const Atom &Astatine;
	static const Atom &Radon;
	static const Atom &Francium;
	static const Atom &Radium;
	static const Atom &Actinium;
	static const Atom &Thorium;		//90
	static const Atom &Protactinium;
	static const Atom &Uranium;
	static const Atom &Neptunium;
	static const Atom &Plutonium;
	static const Atom &Americium;
	static const Atom &Curium;
	static const Atom &Berkelium;
	static const Atom &Californium;
	static const Atom &Einsteinium;
	static const Atom &Fermium;		//100
	static const Atom &Mendelevium;
	static const Atom &Nobelium;
	static const Atom &Lawrencium;
	static const Atom &Rutherfordium;
	static const Atom &Dubnium;
	static const Atom &Seaborgium;
	static const Atom &Bohrium;
	static const Atom &Hassium;
	static const Atom &Meitnerium;
	static const Atom &Darmstadtium;	//110
	static const Atom &Roentgenium;
	static const Atom &Copernicium;
	static const Atom &Nihonium;
	static const Atom &Flerovium;
	static const Atom &Moscovium;
	static const Atom &Livermorium;
	static const Atom &Tennessine;
	static const Atom &Oganesson;

	/** Atoms by symbol. */
	static const Atom &H;
	static const Atom &He;
	static const Atom &Li;
	static const Atom &Be;
	static const Atom &B;
	static const Atom &C;
	static const Atom &N;
	static const Atom &O;
	static const Atom &F;
	static const Atom &Ne;	//10
	static const Atom &Na;
	static const Atom &Mg;
	static const Atom &Al;
	static const Atom &Si;
	static const Atom &P;
	static const Atom &S;
	static const Atom &Cl;
	static const Atom &Ar;
	static const Atom &K;
	static const Atom &Ca;	//20
	static const Atom &Sc;
	static const Atom &Ti;
	static const Atom &V;
	static const Atom &Cr;
	static const Atom &Mn;
	static const Atom &Fe;
	static const Atom &Co;
	static const Atom &Ni;
	static const Atom &Cu;
	static const Atom &Zn;	//30
	static const Atom &Ga;
	static const Atom &Ge;
	static const Atom &As;
	static const Atom &Se;
	static const Atom &Br;
	static const Atom &Kr;
	static const Atom &Rb;
	static const Atom &Sr;
	static const Atom &Y;
	static const Atom &Zr;	//40
	static const Atom &Nb;
	static const Atom &Mo;
	static const Atom &Tc;
	static const Atom &Ru;
	static const Atom &Rh;
	static const Atom &Pd;
	static const Atom &Ag;
	static const Atom &Cd;
	static const Atom &In;
	static const Atom &Sn;	//50
	static const Atom &Sb;
	static const Atom &Te;
	static const Atom &I;
	static const Atom &Xe;
	static const Atom &Cs;
	static const Atom &Ba;
	static const Atom &La;
	static const Atom &Ce;
	static const Atom &Pr;
	static const Atom &Nd;	//60
	static const Atom &Pm;
	static const Atom &Sm;
	static const Atom &Eu;
	static const Atom &Gd;
	static const Atom &Tb;
	static const Atom &Dy;
	static const Atom &Ho;
	static const Atom &Er;
	static const Atom &Tm;
	static const Atom &Yb;	//70
	static const Atom &Lu;
	static const Atom &Hf;
	static const Atom &Ta;
	static const Atom &W;
	static const Atom &Re;
	static const Atom &Os;
	static const Atom &Ir;
	static const Atom &Pt;
	static const Atom &Au;
	static const Atom &Hg;	//80
	static const Atom &Tl;
	static const Atom &Pb;
	static const Atom &Bi;
	static const Atom &Po;
	static const Atom &At;
	static const Atom &Rn;
	static const Atom &Fr;
	static const Atom &Ra;
	static const Atom &Ac;
	static const Atom &Th;	//90
	static const Atom &Pa;
	static const Atom &U;
	static const Atom &Np;
	static const Atom &Pu;
	static const Atom &Am;
	static const Atom &Cm;
	static const Atom &Bk;
	static const Atom &Cf;
	static const Atom &Es;
	static const Atom &Fm;	//100
	static const Atom &Md;
	static const Atom &No;
	static const Atom &Lr;
	static const Atom &Rf;
	static const Atom &Db;
	static const Atom &Sg;
	static const Atom &Bh;
	static const Atom &Hs;
	static const Atom &Mt;
	static const Atom &Ds;	//110
	static const Atom &Rg;
	static const Atom &Cn;
	static const Atom &Nh;
	static const Atom &Fl;
	static const Atom &Mc;
	static const Atom &Lv;
	static const Atom &Ts;
	static const Atom &Og;
private:
	/** Number of protons. */
	unsigned int atomicNumber;

	/** Number of atoms in the periodic table. */
	static constexpr unsigned int totalNumAtoms = 118;

	/** Symbols. */
	static std::string symbols[];

	/** Names. */
	static std::string names[];

	/** Standard weight. */
	static double standardWeights[];

	/** Atoms. */
	static Atom atoms[];
};

inline unsigned int Atom::getAtomicNumber() const{
	return atomicNumber;
}

inline std::string Atom::getSymbol() const{
	return symbols[atomicNumber];
}

inline std::string Atom::getName() const{
	return names[atomicNumber];
}

inline double Atom::getStandardWeight() const{
	return UnitHandler::convertMassDtN(
		standardWeights[atomicNumber],
		UnitHandler::MassUnit::u
	);
}

inline const Atom& Atom::getAtomByNumber(unsigned int atomicNumber){
	if(atomicNumber > totalNumAtoms)
		return atoms[0];
	else
		return atoms[atomicNumber];
}

inline const Atom& Atom::getAtomBySymbol(const std::string &symbol){
	for(unsigned int n = 1; n <= totalNumAtoms; n++)
		if(symbols[n].compare(symbol) == 0)
			return atoms[n];

	return atoms[0];
}

inline const Atom& Atom::getAtomByName(const std::string &name){
	for(unsigned int n = 1; n <= totalNumAtoms; n++)
		if(names[n].compare(name) == 0)
			return atoms[n];

	return atoms[0];
}

};	//End namespace TBTK

#endif
