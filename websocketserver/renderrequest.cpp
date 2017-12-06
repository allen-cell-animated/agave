#include "renderrequest.h"

RenderRequest::RenderRequest(QWebSocket *client, std::vector<Command*> parameters, bool debug) :
	client(client),
	parameters(parameters),
	debug(debug)
{
	this->actualDuration = 0;
	this->estimatedDuration = 10;
}
