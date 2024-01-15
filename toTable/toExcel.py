import xlsxwriter
import datetime

class toExcel:
    def __init__(self, incidence, exposed_values, lambda_values, labels, Data, Predict):
        self.incidence = incidence
        self.exposed_values = exposed_values
        self.lambda_values = lambda_values
        self.labels = labels
        self.Data = Data
        self.Predict = Predict
        self.len = len(labels)
        self.len3 = self.len*3

    def generate(self):
        current_time = str(datetime.datetime.now())
        workbook = xlsxwriter.Workbook(f'./gui/static/excel/{self.incidence}{current_time.replace(" ", "_")}.xlsx')
        worksheet = workbook.add_worksheet("Данные")
        #merge_cells(start_row=x, start_column=1, end_row=x, end_column=4)
        
        bold_сell_format = workbook.add_format({'bold': True, 'align': 'center'})
        cell_format = workbook.add_format({'align': 'center'})
        left_format = workbook.add_format({'align': 'left'})

        self._mainBody(worksheet, bold_сell_format)

        workbook.close()

    def _mainBody(self, worksheet, bold_сell_format, ):
        
        
        for i in range(0, self.len3, 3):
            worksheet.set_column(0, i, 10)
            worksheet.set_column(0, i+1, 10)
            worksheet.merge_range(0, i, 0, i+1, self.labels[i//3], bold_сell_format)
            worksheet.write(1, i, "Модель", bold_сell_format)
            worksheet.write(1, i+1, 'Данные', bold_сell_format)

            data = self.Data[i//3]
            predict = self.Predict[i//3]

            for j in range(len(data)):
                worksheet.write(j+2, i, predict[j])
                worksheet.write(j+2, i+1, data[j])

    def _strains(self):
        pass
    
    def _strain(self):
        pass
