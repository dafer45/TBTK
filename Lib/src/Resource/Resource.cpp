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

/** @file Resource.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Resource.h"
#include "TBTK/TBTKMacros.h"

#include <sstream>

#include <curl/curl.h>

using namespace std;

namespace TBTK{

Resource::Resource(){
}

Resource::~Resource(){
}

static int counter;
void Resource::write(const string &uri){
	//If no "scheme://" is specified, write as normal file.
	size_t position = uri.find("://");
	if(position == string::npos){
		ofstream fout(uri);
		fout << data;
		fout.close();
		return;
	}

	string scheme = uri.substr(0, position);
	if(scheme.compare("file") == 0){
		string filename = uri.substr(position + 3, uri.size() - position - 3);
		ofstream fout(filename);
		fout << data;
		fout.close();
	}
	else{
		TBTKExit(
			"Resource::write()",
			"Scheme '" << scheme << "' not yet supported. Currently"
			<< " only write to local storage is supported.",
			""
		);
	}
}

void Resource::read(const string &uri){
	//If no "scheme://" is specified, read as normal file.
	size_t position = uri.find("://");
	if(position == string::npos){
		ifstream fin(uri);
		TBTKAssert(
			fin.is_open(),
			"Resource::read()",
			"Unable to open file '" << uri << "'.",
			""
		);
		stringstream ss;
		ss << fin.rdbuf();
		data = ss.str();
		fin.close();
		return;
	}

	data.clear();
	CURL *easyHandle = curl_easy_init();
	curl_easy_setopt(easyHandle, CURLOPT_URL, uri.c_str());
	curl_easy_setopt(easyHandle, CURLOPT_WRITEFUNCTION, readCallback);
	curl_easy_setopt(easyHandle, CURLOPT_WRITEDATA, this);

	CURLcode curlCode = curl_easy_perform(easyHandle);
	switch(curlCode){
	case CURLE_OK:
		break;
	case CURLE_FILE_COULDNT_READ_FILE:
	{
		TBTKExit(
			"Resource::read()",
			"Unable to open file.",
			"Check the URI and note that the path should be"
			<< " specified as 'file:///path/to/file', with THREE"
			<< " slashes following 'file:'."
		);
		break;
	}
	default:
	{
		TBTKExit(
			"Resource::read()",
			"Failed to read resource with CURL error code " << curlCode << ".",
			""
		);
	}
	}

	curl_easy_cleanup(easyHandle);
}

size_t Resource::writeCallback(
	void *data,
	size_t size,
	size_t nmemb,
	void *userdata
){
	if(size*nmemb < 1)
		return 0;

	if(counter != 0){
		*(char*)data = ((Resource*)userdata)->data.at(((Resource*)userdata)->data.size() - counter);
		counter--;
		return 1;
	}

	return 0;
}

size_t Resource::readCallback(
	char *data,
	size_t size,
	size_t nmemb,
	void *userdata
){
	for(unsigned int n = 0; n < size*nmemb; n++)
		((Resource*)userdata)->data += data[n];

	return size*nmemb;
}

};	//End of namespace TBTK
