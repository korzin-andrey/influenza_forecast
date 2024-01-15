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

        if self.incidence in ['strain', 'strain_age-group']:
            self._strains(worksheet, bold_сell_format, cell_format, left_format)
        else:
            self._strain(worksheet, bold_сell_format, cell_format, left_format)

        workbook.close()

    def _mainBody(self, worksheet, bold_сell_format):
        
        
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

    def _strains(self, worksheet, bold_сell_format, cell_format, left_format):
        worksheet.merge_range(0, self.len3, 0, self.len3+2, "Доля переболевших", bold_сell_format)
        worksheet.set_column(0, self.len3, 10)
        worksheet.set_column(0, self.len3+1, 10)
        
        for i in range(self.len):
            worksheet.merge_range(i+1, self.len3, i+1, self.len3+1, self.labels[i], left_format)
            worksheet.write(i+1, self.len3+2, self.exposed_values[i], cell_format)

        virul = ['A(H1N1)', 'A(H3N2)', 'B']
        worksheet.merge_range(self.len+2, self.len3, self.len+2, self.len3+1, "Вирулентность", bold_сell_format)
        for i in range(3):
            worksheet.write(self.len+3+i, self.len3, virul[i], left_format)
            worksheet.write(self.len+3+i, self.len3+1, self.lambda_values[i], cell_format)
            

            
    
    def _strain(self, worksheet, bold_сell_format, cell_format, left_format):
        worksheet.merge_range(0, self.len3, 0, self.len3+1, "Доля переболевших", bold_сell_format)
        worksheet.set_column(0, self.len3, 10)
        worksheet.set_column(0, self.len3+1, 10)
        
        for i in range(self.len):
            worksheet.write(i+1, self.len3, self.labels[i], cell_format)
            worksheet.write(i+1, self.len3+1, self.exposed_values[i], cell_format)

        worksheet.merge_range(self.len+2, self.len3, self.len+2, self.len3+1, "Вирулентность", bold_сell_format)
        worksheet.merge_range(self.len+3, self.len3, self.len+3, self.len3+1, self.lambda_values[0], cell_format)
            










