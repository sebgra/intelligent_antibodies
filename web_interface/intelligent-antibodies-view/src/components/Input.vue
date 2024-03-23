<template>
  <div class="input-wrapper">
    <div class="input_title">
      <h3>
        {{ title }}
      </h3>

      Please submit sequence
    </div>
      <div class="input_field-wrapper">
        <input class="input_field" :placeholder="text" @input="input_text">
        <div class='error' v-if="getError.length > 0">
          {{ getError }}
        </div>
      </div>
      <div class="input_fasta-text">
        or select local fasta file
      </div>
      <div class="input_button-wrapper">
        <button class="input_button" @click="find_fasta">
          Explore
        </button>
        <div v-if="selected.length > 0" class="input_button-selected">
          {{ selected }}
        </div>
        <div v-else>
          No file selected
        </div>
      </div>
    </div>
</template>
<script>
 export default {
    name: 'Input',
    data() {
      return {
        current_text: '',
        selected: '',
        error: '',
        aminoAcids: ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'X']
      }
    },
    props: {
      text:{
        type:String,
      },
      title:{
        type:String,
      }
    },
    computed: {
      getError() {
        return this.error
      }
    },
    methods: {
      input_text(event) {
        if (this.rules(event.target.value)) {
          this.error = ''
          this.current_text = event.target.value
          this.emitText(this.current_text)
        } else {
          this.error = event.target.value + ' is not a protein !'
          this.emitError()
        }
      },
      emitError() {
        this.$emit('error')
      },
      emitText(item) {
         this.$emit('update-text', item)
      },
      find_fasta() {
        console.log(find_fasta)
      },
      rules(sequence) {
        const re = new RegExp("^["+ this.aminoAcids.join("")+"]*$")
        return sequence.length === 0 || sequence.match(re)
      }
    }
  }

</script>


<style scoped>
.input_title {
  padding: 5px;
  margin: 5px;
}
.error {
  padding:5px;
  font-size:10;
  color:red;
}
.input-wrapper {
  display: block;
  min-height: 2rem;
  width: 100%;
}
.input_field-wrapper {
  width: 100%;
  padding: 5px
}
.input_fasta-text {
  padding: 5px
}
.input_field {
  width: 100%;
  min-height: 2rem;
}
.input_interaction {
  display: flex;
  justify-content: space-between;
}
.input_button {
  border: none;
  padding: 5px;
  margin-right: 5px
}
.input_button-wrapper {
  display: flex;
  margin: 5px;
  justify-content: left;
}

</style>
