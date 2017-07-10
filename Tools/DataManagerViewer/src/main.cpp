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

/** @package TBTKTools
 *  @file main.cpp
 *  @brief View for displaying Properties.
 *
 *  @author Kristofer Björsnon
 */

#include "DataManagerFrame.h"

#include <complex>

#include "wx/wxprec.h"
#ifndef WX_PRECOMP
#	include "wx/wx.h"
#endif
#include "wx/sizer.h"

using namespace std;
using namespace TBTK;

const complex<double> i(0, 1);

class DataManagerViewer : public wxApp{
public:
	bool OnInit();
private:
};

DECLARE_APP(DataManagerViewer);
IMPLEMENT_APP(DataManagerViewer);

bool DataManagerViewer::OnInit(){
	Streams::openLog();

	wxInitAllImageHandlers();

	DataManagerFrame * window = new DataManagerFrame(nullptr);
	SetTopWindow(window);

//	wxBoxSizer *sizer = new wxBoxSizer(wxHORIZONTAL);
//	window->SetSizer(sizer);

	window->Show(true);

	Streams::closeLog();

	return true;
}
