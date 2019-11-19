#include "TBTK/Exception.h"

#include <iostream>

using namespace std;

namespace TBTK{

Exception::Exception(){
}

Exception::Exception(
	const string& function,
	const string& where,
	const string& message,
	const string& hint
){
	this->function = function;
	this->where = where;
	this->message = message;
	this->hint = hint;
}

Exception::~Exception(){
}

void Exception::print() const{
	cerr << "Error in " << function << "\n";
	cerr << "\t" << message << "\n";
	if(hint.compare("") != 0)
		cerr << "\tHint: " << hint << "\n";
	cerr << "\tWhere: " << where << "\n";
}

const char* Exception::what() const noexcept{
	return message.c_str();
}

};	//End of namespace TBTK
