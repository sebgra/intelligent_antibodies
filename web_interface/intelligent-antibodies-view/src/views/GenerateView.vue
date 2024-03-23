<template>
  <div class="generation">
    <div class="input">
      <InputVue 
        text="Enter an antigene's peptidic sequence"
        title="Antigene input"
        @updateText="updateText"
        @error="setInputError"
      />
    </div>
    <div class="action">
      <Action title="Launch generation" :activated="isValidText&!getInputError" @action="sendRequest"/>
    </div>
    <div class="table-view" v-if="hasDatas">
        <h3 style="margin-bottom:5px">
          Generated Antibodies
        </h3>

      <TableView :datas="getTableDatas"/>
    </div>
    <div v-else-if="getIsLoading">
      <div class="center">
        <img  src="@/assets/loading.gif" alt="Loading" />
      </div>
      <div class="center">
        <h3>Generating data ...</h3>
      </div>
    </div>
    <div v-if="getIsError">
      An error occured: {{ error }}
    </div>
  </div>
</template>

<script>
import InputVue from "../components/Input.vue"
import Action from "../components/Action.vue"
import TableView from "../components/Table.vue"
import api from '@/api/api.js'
export default {
    name: 'GenerateView',
    data() {
      return {
        antigene: '',
        isLoading: false,
        isError:false,
        error:'',
        tableDatas: [],
        inputError: false
      }
    },
    components: {
      InputVue,
      Action,
      TableView
    },
    computed: {
      isValidText: function() {
        return Boolean(this.antigene.length > 0)
      },
      getIsLoading: function() {
        return this.isLoading
      },
      getIsError: function() {
        return this.isError
      },
      hasDatas: function() {
        return this.tableDatas.length > 0
      },
      getTableDatas: function() {
        return this.tableDatas
      },
      getInputError: function() {
        return this.inputError
      }
    },
    methods: {
      updateText(emited) {
        this.antigene = emited
        this.inputError = false
      },
      setInputerror() {
        this.inputError = true
      },
      transpose(table) {
        var new_table = []
        const num_results = table[0].length
        for (var i = 0; i < num_results; i++) {
          var col_values =  table.map(x => x[i])
          new_table.push(col_values)
        }
        return new_table
      },
      parse_result(result) {
        var table = []
        var keys = Object.keys(result)
        keys.forEach(function(key){
          const current_col = result[key]
          var temp = []
          for (const col in Object.keys(current_col)) {
            temp.push(current_col[col])
          }
          table.push(temp)
        });
        return this.transpose(table)
      },
      async sendRequest () {
        try {
          this.isError = false
          this.isLoading = true
          const res = await api.generate.getGenerateData(this.antigene)
          this.isLoading = false
          this.tableDatas = this.parse_result(res)
        } catch (error) {
          this.isLoading = false
          this.error = error
          this.isError = true
        }
      },
    }
  }

</script>

<style>
.generation {
  display: block;
  align-items: center;
  border: 1px solid lightgrey;
  border-radius: 12px; 
  padding: 5px;
  margin-top: 10px
}

.loader {
  margin:auto;
  display: block;
  justify-content: center;
}

.center {
  margin: auto;
  display: flex;
  justify-content: center;
}

.input {
  width: 100%;
}

.action {
  width: 100%;
  margin-bottom:10px;
}

.table-view {
  width: 100%;
  padding: 10px;
}
</style>