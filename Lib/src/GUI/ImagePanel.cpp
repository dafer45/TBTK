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

/** @file ImagePanel.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Streams.h"
#include "TBTK/ImagePanel.h"

using namespace std;

namespace TBTK{

ImagePanel::ImagePanel(
	wxWindow *parent
) : wxPanel(parent){
}

ImagePanel::~ImagePanel(){
}

void ImagePanel::setImage(wxString file, wxBitmapType format){
	image.LoadFile(file, format);
	width = -1;
	height = -1;
//	paintNow();
}

BEGIN_EVENT_TABLE(ImagePanel, wxPanel)

EVT_PAINT(ImagePanel::onPaintEvent)
EVT_SIZE(ImagePanel::onSizeEvent)

END_EVENT_TABLE()

void ImagePanel::onPaintEvent(wxPaintEvent &event){
	wxPaintDC dc(this);
	render(dc);
}

void ImagePanel::onSizeEvent(wxSizeEvent &event){
	Layout();
	Refresh();
	event.Skip();
}

void ImagePanel::paintNow(){
	wxClientDC dc(this);
	render(dc);
}

int counter = 0;
void ImagePanel::render(wxDC &dc){
	if(image.IsOk()){
		int newWidth;
		int newHeight;
		dc.GetSize(&newWidth, &newHeight);

		if(newWidth != width || newHeight != height){
			resizedImage = wxBitmap(
				image.Scale(newWidth, newHeight)
			);
			width = newWidth;
			height = newHeight;
		}

		dc.DrawBitmap(resizedImage, 0, 0, false);
	}
}

};	//End of namespace TBTK
