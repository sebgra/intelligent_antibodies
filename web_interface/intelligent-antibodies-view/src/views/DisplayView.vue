<template>
  <div class="about">
    <div class="display" v-if="$route.params.sequence">
      <div class="display_sequence">
        <div class="title">
          <h3 style="margin:5px; font-weight:700">Antibody's sequence</h3> 
          <h3 style="margin:auto; margin-left:10px"> Length: {{ $route.params.sequence.length }} aa</h3>
        </div>
        <div class="sequence">
          {{ $route.params.sequence }}
        </div>
      </div>
      <div class="display_structure">
        <div class="title">
          <h3 style="margin:5px; font-weight:700">3D structure</h3> 
          <h3 style="margin:auto; margin-left:10px"> From PDB closest sequence {{ getId }}</h3>
        </div>
      </div>
      <div v-if="getIsLoaded">
        <protein :protId="getId"/>
        <h6 style="margin: auto;text-align: center;">PV © Copyright 2013-2015, Marco Biasini, DOI: 10.5281/zenodo.592834</h6>
      </div>
      <div v-if="!getIsLoaded">
          <div class="center">
            <img  src="@/assets/loading.gif" alt="Loading"  />
          </div>
          <div class="center">
            <h3>Retrieving structure ...</h3>
          </div>
        </div>
    </div>
    <div v-else class="display_retrieve">
      <InputView
        text="Enter an antigene's peptidic sequence"
        title="Antibody input" 
        @updateText="updateText"
        @error="setInputError"
      />
      <div class="action">
        <Action title="Retrieve antibody data" :activated="antibody.length > 0 &!getInputError" @action="gotodisplay"/>
      </div>
    </div>
  </div>
</template>
<script>
import Action from '../components/Action.vue'
import InputView from '../components/Input.vue'
import Protein from '../components/Protein.vue'
import api from '@/api/api.js'

export default {
  name:'Display',
  components: {
    InputView,
    Action,
    Protein
  },
  data() {
    return {
      antibody: '',
      id: '(Loading...)',
      isLoaded: false,
      pdbFile: ''
    }
  },
  computed: {
    getIsLoaded() {
      return this.isLoaded
    },
    getId() {
      return this.id
    }
  },
  async mounted() {
    if (this.$route.params.sequence) {
      this.antibody = this.$route.params.sequence
      this.id = await api.structure.getId(this.antibody)
      this.pdbFile = await api.structure.getPDB(this.id)
      this.isLoaded = true
    }
  },
  methods: {
    updateText(text) {
      this.error = false
      this.antibody = text
    },
    setInputError() {
      this.error = true
    },
    gotodisplay(event) {
      this.$router.push({ path: '/display/' + this.antibody})
    },
  }
}
</script>

<style>
.display_sequence {
  display: block;
  align-items: center;
  background-color: #38a3a5;
  border-radius: 12px; 
  padding: 5px;
  margin-top: 10px;
  color: white
}
.display_retrieve {
  display: block;
  align-items: center;
  border: 1px solid lightgrey;
  border-radius: 12px; 
  padding: 5px;
  margin-top: 10px
}
.sequence {
  background-color: white;
  height: 10rem;
  overflow-wrap: break-word;
  overflow: auto;
  border-radius: 5px;
  color: #3f5060;
  padding: 5px
}
.title {
  display:flex;
  margin-bottom:10px;
  justify-content: center;
}
.display_structure {
  display: block;
  align-items: center;
  background-color: #38a3a5;
  border-radius: 12px; 
  padding: 5px;
  margin-top: 10px;
  color: white
}

.center {
  margin: auto;
  display: flex;
  justify-content: center;
}

</style>
