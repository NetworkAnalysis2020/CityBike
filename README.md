## CityBike

Study project for University of Helsinki course Network Analysis (Spring 2020).

### Bicycle data

Default data used by the methods covers all bicycle data between April and October 2019.
This is however too heavy dataload for two methods, draw_stations_and_edges_to_map and
predict_destination in citybike.py. For these it is recommended to use only data for one month. 

To change this, in import_data-method in citybike.py change filepath of the imported file from "./data/*.csv" into one of the single csv:s in data-folder, for example "./data/2019-05.csv"



[Project report](https://docs.google.com/document/d/1As3tlzw6EQEs-u93LUYdANUYLAFR6rSOErFXA1ypWa4/edit)
[Project presentation](https://docs.google.com/presentation/d/1mGkY2bHRbybF39NytH6qcksqoLbSrZyXeg7TnCXHtfo/edit#slide=id.)

