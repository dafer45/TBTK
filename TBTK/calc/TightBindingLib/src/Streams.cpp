/** @file Streams.cpp
 *
 *  @author Kristofer Björnson
 */

#include "../include/Streams.h"

#include <iostream>

using namespace std;

namespace TBTK{
namespace Util{

ostream Streams::out(new ForkBuffer(&cout, &Streams::log));
ostream Streams::log(&nullBuffer);
ostream Streams::err(new ForkBuffer(&cout, &Streams::log));
Streams::NullBuffer Streams::nullBuffer;
ostream Streams::null(&nullBuffer);

int Streams::NullBuffer::overflow(int c){
	return c;
}

/*void Streams::muteOut(){
	out.setStream(&null);
}

void Streams::muteLog(){
	log.setStream(&null);
}

void Streams::muteErr(){
	err.setStream(&null);
}*/

Streams::ForkBuffer::ForkBuffer(basic_ostream<char, char_traits<char>> *ostream1, basic_ostream<char, char_traits<char>> *ostream2) :
	ostream1(ostream1),
	ostream2(ostream2){
}

int Streams::ForkBuffer::overflow(int c){
	if(ostream1)
		*ostream1 << (char)c;
	if(ostream2)
		*ostream2 << (char)c;
	return c;
}

};	//End of namespace Util
};	//End of namespace TBTK
