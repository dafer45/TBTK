/** @file Streams.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Streams.h"

#include <iostream>

using namespace std;

namespace TBTK{
namespace Util{

ostream &Streams::out = cout;
ostream &Streams::log = cout;
ostream &Streams::err = cout;

};	//End of namespace Util
};	//End of namespace TBTK
